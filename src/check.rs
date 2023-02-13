use std::collections::{HashMap, hash_map::Entry};

use z3::{Context, Config, DatatypeSort, DatatypeBuilder, ast::{Dynamic, Datatype, Bool, Ast as Z3Ast}, Optimize, SatResult, Model};

use super::parse::{Ast, AstVariants, Type};

struct Z3State<'a> {
    context: &'a Context,
    type_datatype: DatatypeSort<'a>,
    any_z3: Dynamic<'a>,
    number_z3: Dynamic<'a>,
    bool_z3: Dynamic<'a>,
}

impl<'a> Z3State<'a> {
    fn is_variant(&self, i: usize, model: &Model, e: &Dynamic) -> bool {
        model
            .eval(&self.type_datatype.variants[i].tester.apply(&[e]).as_bool().unwrap(), true)
            .unwrap()
            .as_bool()
            .unwrap()
    }

    fn is_any(&self, model: &Model, e: &Dynamic) -> bool {
        self.is_variant(0, model, e)
    }

    fn is_number(&self, model: &Model, e: &Dynamic) -> bool {
        self.is_variant(1, model, e)
    }

    fn is_bool(&self, model: &Model, e: &Dynamic) -> bool {
        self.is_variant(2, model, e)
    }

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

struct State<'a> {
    _config: Config,
    z3: Z3State<'a>,
    solver: Optimize<'a>,
    vars: HashMap<u32, Dynamic<'a>>,
    metavar_index: u32,
}

impl<'a> State<'a> {
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

    fn z3_bool(&self, b: bool) -> Bool<'a> {
        Bool::from_bool(self.z3.context, b)
    }

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

    fn ground(&self, _t: &Type) -> Bool<'a> {
        // temporary
        self.z3_bool(true)
    }

    fn coerce(&mut self, t1: Type, t2: Type, ast: &mut Ast) {
        let t1_z3 = self.type_to_z3_sort(&t1);
        let t2_z3 = self.type_to_z3_sort(&t2);
        self.solver.assert_soft(&t1_z3._eq(&t2_z3), 1, None);
        ast.type_ = t1;
        ast.coercion = Some(t2);
    }

    fn weaken(&mut self, t1: Type, ast: &mut Ast, phi1: Bool<'a>) -> (Type, Bool<'a>) {
        let alpha = self.next_metavar();
        let coerce_case = self.type_to_z3_sort(&alpha)._eq(&self.z3.any_z3) & self.ground(&t1);
        let dont_coerce_case = self.type_to_z3_sort(&t1)._eq(&self.type_to_z3_sort(&alpha));
        self.coerce(t1, alpha.clone(), ast);
        (alpha, phi1 & (coerce_case | dont_coerce_case))
    }

    #[must_use]
    fn strengthen(&mut self, t1: Type, t2: Type, ast: &mut Ast) -> Bool<'a> {
        let coerce_case = self.type_to_z3_sort(&t1)._eq(&self.z3.any_z3) & self.ground(&t2);
        // we don't care about putting an ID coercion, that's fine
        let t1_z3 = self.type_to_z3_sort(&t1);
        let t2_z3 = self.type_to_z3_sort(&t2);
        let dont_coerce_case = t1_z3._eq(&t2_z3);
        self.coerce(t1, t2, ast);
        coerce_case | dont_coerce_case
    }

    fn solve_model(&self, model: Model) -> HashMap<u32, Type> {
        let mut result = HashMap::new();
        for (x, x_ast) in self.vars.iter() {
            let x_val_ast = model.eval(x_ast, true).expect("evaluating metavar");
            result.insert(*x, self.z3.z3_to_type(&model, x_val_ast));
        }
        result
    }

    fn annotate_type(&self, model_result: &HashMap<u32, Type>, t: &mut Type) {
        if let Type::Metavar(i) = t {
            if let Some(s) = model_result.get(i) {
                *t = s.clone();
            }
        }
    }

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

