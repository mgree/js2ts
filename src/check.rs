use std::collections::{HashMap, hash_map::Entry};

use z3::{Context, Config, DatatypeSort, DatatypeBuilder, ast::{Dynamic, Datatype, Bool, Ast as Z3Ast}, Optimize, SatResult, Model};

use super::parse::{Ast, AstVariants, Type};

/// Contains various Z3 structs such as the context and type datatype, as well as the variants of the type datatype.
struct Z3State<'a> {
    context: &'a Context,
    type_datatype: DatatypeSort<'a>,
    any_z3: Dynamic<'a>,
    number_z3: Dynamic<'a>,
    bool_z3: Dynamic<'a>,
}

impl<'a> Z3State<'a> {
    /// Checks if a given dynamic value is the given variant.
    fn is_variant(&self, i: usize, model: &Model, e: &Dynamic) -> bool {
        model
            .eval(&self.type_datatype.variants[i].tester.apply(&[e]).as_bool().unwrap(), true)
            .unwrap()
            .as_bool()
            .unwrap()
    }

    /// Checks if the given dynamic value is an any type.
    fn is_any(&self, model: &Model, e: &Dynamic) -> bool {
        self.is_variant(0, model, e)
    }

    /// Checks if the given dynamic value is a number type.
    fn is_number(&self, model: &Model, e: &Dynamic) -> bool {
        self.is_variant(1, model, e)
    }

    /// Checks if the given dynamic value is a bool type.
    fn is_bool(&self, model: &Model, e: &Dynamic) -> bool {
        self.is_variant(2, model, e)
    }

    /// Converts a Z3 datatype type into a [`Type`].
    pub fn z3_to_type(&self, model: &'a Model, e: Dynamic) -> Type {
        if self.is_any(model, &e) {
            Type::Any
        } else if self.is_number(model, &e) {
            Type::Number
        } else if self.is_bool(model, &e) {
            Type::Bool
        } else {
            panic!("missing case in z3_to_typ");
        }
    }
}

/// Contains various helper functions to migrate untyped Javascript to typed Typescript.
struct State<'a> {
    _config: Config,
    z3: Z3State<'a>,
    solver: Optimize<'a>,
    vars: HashMap<u32, Dynamic<'a>>,
    metavar_index: u32,
}

impl<'a> State<'a> {
    /// Creates a new [`State`].
    fn new(context: &'a Context, config: Config) -> Self {
        let type_datatype = State::type_datatype(context);
        Self {
            _config: config,
            z3: Z3State {
                context,
                any_z3: type_datatype.variants[0].constructor.apply(&[]),
                number_z3: type_datatype.variants[1].constructor.apply(&[]),
                bool_z3: type_datatype.variants[2].constructor.apply(&[]),
                type_datatype,
            },
            solver: Optimize::new(context),
            vars: HashMap::new(),
            metavar_index: 0,
        }
    }

    /// Converts a boolean into a Z3 boolean.
    fn z3_bool(&self, b: bool) -> Bool<'a> {
        Bool::from_bool(self.z3.context, b)
    }

    /// Creates the Z3 datatype that represents a [`Type`].
    fn type_datatype(cxt: &Context) -> DatatypeSort<'_> {
        DatatypeBuilder::new(cxt, "Type")
            .variant("Any", vec![])
            .variant("Number", vec![])
            .variant("Bool", vec![])
            /*
            .variant(
                "Arr",
                vec![
                    ("arg", DatatypeAccessor::Datatype("Typ".into())),
                    ("ret", DatatypeAccessor::Datatype("Typ".into())),
                ],
            )
            .variant(
                "List",
                vec![("lt", DatatypeAccessor::Datatype("Typ".into()))],
            )
            .variant(
                "Pair",
                vec![
                    ("t1", DatatypeAccessor::Datatype("Typ".into())),
                    ("t2", DatatypeAccessor::Datatype("Typ".into())),
                ],
            )
            .variant(
                "Box",
                vec![("bt", DatatypeAccessor::Datatype("Typ".into()))],
            )
            .variant("Unit", vec![])
            .variant(
                "Vect",
                vec![("vt", DatatypeAccessor::Datatype("Typ".into()))],
            )
            .variant("Float", vec![])
            .variant("Char", vec![])
        */
            .finish()
    }

    /// Generates constraints for each [`Ast`] node.
    fn solve_helper(&mut self, ast: &mut Ast) -> (Type, Bool<'a>) {
        if let Type::Any = ast.type_ {
            ast.type_ = self.next_metavar();
        }

        match &mut ast.ast {
            AstVariants::Number(_)
            | AstVariants::Boolean(_) => {
                self.weaken(ast.type_.clone(), ast, self.z3_bool(true))
            }

            AstVariants::Binary { .. } => todo!(),
            AstVariants::Unary { .. } => todo!(),

            AstVariants::Trinary { cond, then, elsy } => {
                let (t1, phi1) = self.solve_helper(&mut **cond);
                let (t2, phi2) = self.solve_helper(&mut **then);
                let (t3, phi3) = self.solve_helper(&mut **elsy);
                let phi4 = self.strengthen(t1, Type::Bool, &mut **cond) & self.type_to_z3_sort(&t2)._eq(&self.type_to_z3_sort(&t3)) & self.type_to_z3_sort(&t2)._eq(&self.type_to_z3_sort(&ast.type_));
                (t2, phi1 & phi2 & phi3 & phi4)
            }
        }
    }

    /// Converts a type into a variant of the Z3 datatype that represents a type.
    fn type_to_z3_sort(&mut self, type_: &Type) -> Dynamic<'a> {
        match type_ {
            Type::Any => self.z3.any_z3.clone(),
            Type::Number => self.z3.number_z3.clone(),
            Type::Bool => self.z3.bool_z3.clone(),
            Type::Metavar(n) => {
                match self.vars.entry(*n) {
                    Entry::Occupied(v) => v.get().clone(),

                    Entry::Vacant(v) => {
                        let t = Datatype::fresh_const(
                            self.z3.context,
                            &type_.to_string(),
                            &self.z3.type_datatype.sort,
                        );
                        let x = Dynamic::from_ast(&t);
                        v.insert(x.clone());
                        x
                    }
                }
            }
        }
    }

    /// Provided a type, generate constraints that the type has any in all of
    /// its negative forms. The function is more weak / general than it could be
    /// due to the difficulties with z3.
    ///
    /// For example, if t has type * -> int, that type is safe to
    /// coerce to any (with wrapping). However, because z3 cannot produce
    /// recursive constraints, and the type * -> (int -> int) is forbidden,
    /// ground is forced to produce the constraint that t has type *
    /// -> *.
    ///
    /// Note that anything that can be mutated is negative.
    ///
    /// One might think that lists are a special case: because lists are
    /// immutable they have no negative positions. However, imagine a function that
    /// is stored in a list. It is inferred to be int -> int, however after being
    /// pulled out of the list it is called with a bool. This is incorrect. We
    /// might say, lists must hold ground types, rather than
    /// any! And you would be right, but notice that we have now produced a
    /// recursive constraint which z3 does not support.
    ///
    /// ground t = is_arr(t) => t = any -> any
    ///                    && is_list(t) => t = list any
    ///                    && is_box(t) => t = box any
    ///                    && is_vect(t) => t = vect any
    fn ground(&self, _t: &Type) -> Bool<'a> {
        // temporary
        self.z3_bool(true)
    }

    /// Adds a coercion to an [`Ast`].
    fn coerce(&mut self, t1: Type, t2: Type, ast: &mut Ast) {
        let t1_z3 = self.type_to_z3_sort(&t1);
        let t2_z3 = self.type_to_z3_sort(&t2);
        self.solver.assert_soft(&t1_z3._eq(&t2_z3), 1, None);
        ast.type_ = t1;
        ast.coercion = Some(t2);
    }

    /// (α, weaken'(t1, α, exp) & phi1) where weaken'(t1, t2, exp) =
    ///
    /// Modifies `exp` in place to corce from t1 to t2. Generates a constraint
    /// that they are already equal, or t2 is any and t1 is
    /// negative-any. Caller's responsibility to ensure typ(exp) = t1
    ///
    /// In other words, the constraint is that t1 and t2 are dynamically
    /// consistent, the type doesn't strengthen, and the coercion does not lose
    /// track of important type information.
    ///
    /// This is always safe, so it happens on all expressions.
    ///
    /// The peculiarities of this signature are because weaken should occur
    /// on every expression that may have a different type than any of its
    /// sub-expressions and NEVER otherwise. Therefore it is easy to call
    /// self.weaken(true_typ, whole_exp, other_constraints) at the end of a match
    /// arm in cgen
    ///
    /// ----------------------------------------------
    /// Γ ⊢ e: T => coerce(T, α, e), α, φ
    ///             && T = α || (α = any && ground(T))      |> weaken'
    fn weaken(&mut self, t1: Type, ast: &mut Ast, phi1: Bool<'a>) -> (Type, Bool<'a>) {
        let alpha = self.next_metavar();
        let coerce_case = self.type_to_z3_sort(&alpha)._eq(&self.z3.any_z3) & self.ground(&t1);
        let dont_coerce_case = self.type_to_z3_sort(&t1)._eq(&self.type_to_z3_sort(&alpha));
        self.coerce(t1, alpha.clone(), ast);
        (alpha, phi1 & (coerce_case | dont_coerce_case))
    }

    /// Modifies `exp` in place to coerce from t1 to t2. Generates a
    /// constraint that T_1 must be any and T_2 must be negative-any, or they are
    /// already equal. Caller's responsibility to ensure typ(exp) = t1
    ///
    /// In other words, the constraint is that t1 and t2 are dynamically
    /// consistent, the type doesn't weaken, and the coercion is reasonable.
    ///
    /// Because this can cause dynamic errors, **this should only be used
    /// at elimination forms** in order to be safe!
    ///
    /// T_1 = T_2 || (T_1 = any && ground(t2))
    #[must_use]
    fn strengthen(&mut self, t1: Type, t2: Type, ast: &mut Ast) -> Bool<'a> {
        let coerce_case = self.type_to_z3_sort(&t1)._eq(&self.z3.any_z3) & self.ground(&t2);
        let dont_coerce_case = self.type_to_z3_sort(&t1)._eq(&self.type_to_z3_sort(&t2));
        self.coerce(t1, t2, ast);
        coerce_case | dont_coerce_case
    }

    /// Converts the model into a map from the metavar to the actual type.
    fn solve_model(&self, model: Model) -> HashMap<u32, Type> {
        let mut result = HashMap::new();
        for (x, x_ast) in self.vars.iter() {
            let x_val_ast = model.eval(x_ast, true).expect("evaluating metavar");
            result.insert(*x, self.z3.z3_to_type(&model, x_val_ast));
        }
        result
    }

    /// Replaces a metavariable with the type Z3 found satisfactory.
    fn annotate_type(&self, model_result: &HashMap<u32, Type>, t: &mut Type) {
        if let Type::Metavar(i) = t {
            if let Some(s) = model_result.get(i) {
                *t = s.clone();
            } else {
                // This probably isn't reached but just in case.
                *t = Type::Any;
            }
        }
    }

    /// Annotates the [`Ast`] with the types that Z3 deemed most appropriate.
    fn annotate(&self, model_result: &HashMap<u32, Type>, ast: &mut Ast) {
        self.annotate_type(model_result, &mut ast.type_);
        if let Some(t) = &mut ast.coercion {
            self.annotate_type(model_result, t);
        }

        match &ast.coercion {
            Some(t) if *t == ast.type_ => {
                ast.coercion = None;
            }

            _ => (),
        }

        match &mut ast.ast {
            AstVariants::Number(_)
            | AstVariants::Boolean(_) => (),

            AstVariants::Binary { left, right, .. } => {
                self.annotate(model_result, &mut **left);
                self.annotate(model_result, &mut **right);
            }

            AstVariants::Unary { value, .. } => {
                self.annotate(model_result, &mut **value);
            }

            AstVariants::Trinary { cond, then, elsy } => {
                self.annotate(model_result, &mut **cond);
                self.annotate(model_result, &mut **then);
                self.annotate(model_result, &mut **elsy);
            }
        }
    }

    /// Creates the next metavariable type.
    fn next_metavar(&mut self) -> Type {
        let next = self.metavar_index;
        self.metavar_index += 1;
        Type::Metavar(next)
    }
}

/// Applies type migration to the generated list of [`Ast`]s.
pub fn solve(asts: &mut [Ast]) -> Result<(), String> {
    let config = Config::new();
    let context = Context::new(&config);
    let mut state = State::new(&context, config);

    let mut phi = state.z3_bool(true);
    for ast in asts.iter_mut() {
        let (_, p) = state.solve_helper(ast);
        phi &= p;
    }

    state.solver.assert(&phi);
    match state.solver.check(&[]) {
        SatResult::Unsat => return Err("unsat (context)".to_string()),
        SatResult::Unknown => panic!("unknown from Z3 -- very bad"),
        SatResult::Sat => (),
    }
    let model = state.solver.get_model().expect("model not available");
    let result = state.solve_model(model);

    for ast in asts {
        state.annotate(&result, ast);
    }

    Ok(())
}

