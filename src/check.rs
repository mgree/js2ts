use std::collections::{hash_map::Entry, HashMap};

use z3::{
    ast::{Ast as Z3Ast, Bool, Datatype, Dynamic},
    Config, Context, DatatypeBuilder, DatatypeSort, Model, Optimize, SatResult,
};

use super::parse::{Ast, AstNode, Type};

/// Contains various Z3 structs such as the context and type datatype, as well as the variants of the type datatype.
struct Z3State<'a> {
    context: &'a Context,
    type_datatype: DatatypeSort<'a>,
    any_z3: Dynamic<'a>,
    number_z3: Dynamic<'a>,
    bool_z3: Dynamic<'a>,
    unit_z3: Dynamic<'a>,
}

impl<'a> Z3State<'a> {
    /// Checks if a given dynamic value is the given variant.
    fn is_variant(&self, i: usize, model: &Model, e: &Dynamic) -> bool {
        model
            .eval(
                &self.type_datatype.variants[i]
                    .tester
                    .apply(&[e])
                    .as_bool()
                    .unwrap(),
                true,
            )
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

    /// Checks if the given dynamic value is a unit type.
    fn is_unit(&self, model: &Model, e: &Dynamic) -> bool {
        self.is_variant(3, model, e)
    }

    /// Converts a Z3 datatype type into a [`Type`].
    pub fn z3_to_type(&self, model: &'a Model, e: Dynamic) -> Type {
        if self.is_any(model, &e) {
            Type::Any
        } else if self.is_number(model, &e) {
            Type::Number
        } else if self.is_bool(model, &e) {
            Type::Bool
        } else if self.is_unit(model, &e) {
            Type::Unit
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
                unit_z3: type_datatype.variants[3].constructor.apply(&[]),
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
            .variant("Unit", vec![])
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
                .variant(
                    "Vect",
                    vec![("vt", DatatypeAccessor::Datatype("Typ".into()))],
                )
            */
            .finish()
    }

    /// Generates constraints for each [`Ast`] node.
    fn generate_constraints(
        &mut self,
        env: &mut Vec<(String, Type)>,
        ast: &mut Ast,
    ) -> (Type, Bool<'a>) {
        match &mut ast.ast {
            // ---------------------------
            // Γ ⊢ lit => coerce(lit.typ(), α, lit), α, weaken(lit.typ(), α)
            v @ (AstNode::Number(_) | AstNode::Boolean(_)) => {
                let type_ = match v {
                    AstNode::Number(_) => Type::Number,
                    AstNode::Boolean(_) => Type::Bool,
                    _ => unreachable!(),
                };

                self.weaken(type_, ast, self.z3_bool(true))
            }

            AstNode::Binary { .. } => todo!(),
            AstNode::Unary { .. } => todo!(),

            // Γ ⊢ e_1 => T_1, φ_1
            // Γ ⊢ e_2 => T_2, φ_2
            // Γ ⊢ e_3 => T_3, φ_3
            // ----------------------------------------------
            // Γ ⊢ if e_1 then e_2 else e_3 => if coerce(T_1, bool, e_1) then e_2 else e_3, T_2,
            //                                 φ_1 && φ_2 && φ_3 &&
            //                                 strengthen(T_1, bool) && T_2 = T_3
            AstNode::Ternary { cond, then, elsy } => {
                let (t1, phi1) = self.generate_constraints(env, &mut **cond);
                let (t2, phi2) = self.generate_constraints(env, &mut **then);
                let (t3, phi3) = self.generate_constraints(env, &mut **elsy);
                let phi4 = self.strengthen(t1, Type::Bool, &mut **cond)
                    & self.type_to_z3_sort(&t2)._eq(&self.type_to_z3_sort(&t3));
                (t2, phi1 & phi2 & phi3 & phi4)
            }

            AstNode::Coercion { .. } => unreachable!(
                "`AstNode::Coercion` is inserted by the migrator and never found in source"
            ),

            // Γ ⊢ e1 => T_1, φ_1
            // Γ,x:T_1 ⊢ e2 => T_2, φ_2
            // ---------------------------------------
            // Γ ⊢ let x = e1 in e2 => let x = e1 in e2, T_2, φ_1 && φ_2
            AstNode::Declare { vars } => {
                let mut phi = self.z3_bool(true);
                for (var, init) in vars.iter_mut() {
                    if let Some(init) = init {
                        let (type_, phi2) = self.generate_constraints(env, init);
                        env.push((var.clone(), type_));
                        phi &= phi2;
                    }
                }
                (Type::Unit, phi)
            }

            // ---------------------------
            // Γ ⊢ x => x, Γ(x), true
            AstNode::Identifier(var) => {
                let type_ = env
                    .iter()
                    .rev()
                    .find(|(v, _)| v == var)
                    .map(|(_, t)| t.clone())
                    .expect("todo: error handling for bad variables");
                (type_, self.z3_bool(true))
            }

            // Γ,x:T_1 ⊢ e_1 => T_2, φ_1
            // ----------------------------------------------
            // Γ,x:T_1 ⊢ x = e_1 => T_2, φ_1 && T_1 = T_2
            AstNode::Assign { var, expr } => {
                let (t1, phi1) = self.generate_constraints(env, &mut **expr);
                let phi2 =
                    if let Some(t2) = env.iter().rev().find(|(v, _)| v == var).map(|(_, t)| t) {
                        self.type_to_z3_sort(&t1)._eq(&self.type_to_z3_sort(t2))
                    } else {
                        env.push((var.clone(), t1.clone()));
                        self.z3_bool(true)
                    };
                (t1, phi1 & phi2)
            }

            AstNode::Block(block) => {
                let mut phi = self.z3_bool(true);
                for stat in block {
                    let (_, p) = self.generate_constraints(env, stat);
                    phi &= p;
                }

                (Type::Unit, phi)
            }
        }
    }

    /// Converts a type into a variant of the Z3 datatype that represents a type.
    fn type_to_z3_sort(&mut self, type_: &Type) -> Dynamic<'a> {
        match type_ {
            Type::Any => self.z3.any_z3.clone(),
            Type::Number => self.z3.number_z3.clone(),
            Type::Bool => self.z3.bool_z3.clone(),
            Type::Unit => self.z3.unit_z3.clone(),

            Type::Metavar(n) => match self.vars.entry(*n) {
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
            },
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

        // TODO: better way of doing this
        let mut tmp = Ast {
            ast: AstNode::Number(0.0),
            span: Default::default(),
        };
        std::mem::swap(&mut tmp, ast);
        *ast = Ast {
            ast: AstNode::Coercion {
                expr: Box::new(tmp),
                source_type: t1,
                dest_type: t2,
            },
            span: Default::default(),
        };
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
        match &mut ast.ast {
            AstNode::Number(_) | AstNode::Boolean(_) | AstNode::Identifier(_) => (),

            AstNode::Binary { left, right, .. } => {
                self.annotate(model_result, &mut **left);
                self.annotate(model_result, &mut **right);
            }

            AstNode::Unary { value, .. } => {
                self.annotate(model_result, &mut **value);
            }

            AstNode::Ternary { cond, then, elsy } => {
                self.annotate(model_result, &mut **cond);
                self.annotate(model_result, &mut **then);
                self.annotate(model_result, &mut **elsy);
            }

            AstNode::Coercion {
                expr,
                source_type,
                dest_type,
            } => {
                self.annotate(model_result, &mut **expr);
                self.annotate_type(model_result, source_type);
                self.annotate_type(model_result, dest_type);
                if source_type == dest_type {
                    *ast = (**expr).clone();
                }
            }

            AstNode::Declare { vars } => {
                for (_var, init) in vars {
                    if let Some(init) = init {
                        self.annotate(model_result, init);
                    }
                }
            }

            AstNode::Assign { expr, .. } => self.annotate(model_result, &mut **expr),

            AstNode::Block(block) => {
                for stat in block {
                    self.annotate(model_result, stat);
                }
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
    let mut env = Vec::new();
    for ast in asts.iter_mut() {
        let (_, p) = state.generate_constraints(&mut env, ast);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;
    use crate::typecheck::typecheck;

    #[test]
    fn number() {
        let mut v = parse_helper("2");
        solve(&mut v).expect("should not fail");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].to_string(), "2".to_string());
        typecheck(&v).expect("should not fail");
    }

    #[test]
    fn boolean() {
        let mut v = parse_helper("true");
        solve(&mut v).expect("should not fail");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].to_string(), "true".to_string());
        typecheck(&v).expect("should not fail");
    }

    #[test]
    fn ternary_same() {
        let mut v = parse_helper("true ? 2 : 3");
        solve(&mut v).expect("should not fail");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].to_string(), "(true ? 2 : 3)".to_string());
        typecheck(&v).expect("should not fail");
    }

    #[test]
    fn ternary_diff() {
        let mut v = parse_helper("false ? 2 : false");
        solve(&mut v).expect("should not fail");
        assert_eq!(v.len(), 1);
        assert_eq!(
            v[0].to_string(),
            "(false ? (2 : any) : (false : any))".to_string()
        );
        typecheck(&v).expect("should not fail");
    }

    #[test]
    fn ternary_same_cond_not_bool() {
        let mut v = parse_helper("2 ? true : false");
        solve(&mut v).expect("should not fail");
        assert_eq!(v.len(), 1);
        assert_eq!(
            v[0].to_string(),
            "(((2 : any) : bool) ? true : false)".to_string()
        );
        typecheck(&v).expect("should not fail");
    }

    #[test]
    fn ternary_diff_cond_not_bool() {
        let mut v = parse_helper("2 ? false : 3");
        solve(&mut v).expect("should not fail");
        assert_eq!(v.len(), 1);
        assert_eq!(
            v[0].to_string(),
            "(((2 : any) : bool) ? (false : any) : (3 : any))".to_string()
        );
        typecheck(&v).expect("should not fail");
    }

    #[test]
    fn assignment() {
        let mut v = parse_helper("x = 2");
        solve(&mut v).expect("should not fail");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].to_string(), "(x#0 = 2)");
        typecheck(&v).expect("should not fail");
    }

    #[test]
    fn declaration() {
        let mut v = parse_helper("var x = 2");
        solve(&mut v).expect("should not fail");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].to_string(), "var x#0 = 2;");
        typecheck(&v).expect("should not fail");
    }

    #[test]
    fn declare_any() {
        let mut v = parse_helper("var x\nx = 2\nx ? 2 : 3");
        solve(&mut v).expect("should not fail");
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].to_string(), "var x#0;");
        assert_eq!(v[1].to_string(), "(x#0 = (2 : any))");
        assert_eq!(v[2].to_string(), "((x#0 : bool) ? 2 : 3)");
        typecheck(&v).expect("should not fail");
    }
}
