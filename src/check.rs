use std::collections::{HashMap, hash_map::Entry};

use z3::{Context, Config, DatatypeSort, DatatypeBuilder, ast::{Dynamic, Datatype}};

use super::parse::{Ast, AstVariants, Type};

struct Z3State<'a> {
    context: &'a Context,
    type_datatype: DatatypeSort<'a>,
    number_z3: Dynamic<'a>,
    bool_z3: Dynamic<'a>,
    any_z3: Dynamic<'a>,
}

struct State<'a> {
    config: Config,
    z3: Z3State<'a>,
    vars: HashMap<u32, Dynamic<'a>>,
    metavar_index: u32,
}

impl<'a> State<'a> {
    fn new(context: &'a Context, config: Config) -> Self {
        let type_datatype = State::type_datatype(context);
        Self {
            config,
            z3: Z3State {
                context,
                number_z3: type_datatype.variants[0].constructor.apply(&[]),
                bool_z3: type_datatype.variants[1].constructor.apply(&[]),
                any_z3: type_datatype.variants[2].constructor.apply(&[]),
                type_datatype,
            },
            vars: HashMap::new(),
            metavar_index: 0,
        }
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

    fn solve_helper(&mut self, ast: &mut Ast) {
        if let Type::Any = ast.type_ {
            ast.type_ = self.next_metavar();
        }

        match &ast.ast {
            AstVariants::Number(_)
            | AstVariants::Boolean(_) => {
                
            }

            AstVariants::Binary { .. } => todo!(),
            AstVariants::Unary { .. } => todo!(),
            AstVariants::Trinary { .. } => todo!(),
        }
    }

    fn type_to_z3_sort(&mut self, type_: Type) -> Dynamic<'a> {
        match type_ {
            Type::Any => self.z3.any_z3.clone(),
            Type::Number => self.z3.number_z3.clone(),
            Type::Bool => self.z3.bool_z3.clone(),
            Type::Metavar(n) => {
                match self.vars.entry(n) {
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

    fn next_metavar(&mut self) -> Type {
        let next = self.metavar_index;
        self.metavar_index += 1;
        Type::Metavar(next)
    }
}

/// Applies type migration to the generated list of [`Ast`]s.
pub fn solve(asts: &mut [Ast]) {
    let config = Config::new();
    let context = Context::new(&config);
    let mut state = State::new(&context, config);

    for ast in asts.iter_mut() {
        state.solve_helper(ast);
    }
}

