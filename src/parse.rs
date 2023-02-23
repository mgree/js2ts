use std::fmt::Display;

use swc_common::Span;
use swc_ecma_ast::{BinaryOp, Decl, Expr, Lit, ModuleItem, Pat, PatOrExpr, Stmt, UnaryOp};

/// Represents a type.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Type {
    /// The any type (star type in TypeWhich). This is the default type.
    #[default]
    Any,

    /// A metavariable type. This is used in constraint generation. If this type exists after type migration, that is a bug.
    Metavar(u32),

    /// A number type.
    Number,

    /// A boolean type.
    Bool,

    /// A unit type (ie, void).
    Unit,
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Any => write!(f, "any"),
            Type::Number => write!(f, "number"),
            Type::Bool => write!(f, "bool"),
            Type::Metavar(n) => write!(f, "${}", n),
            Type::Unit => write!(f, "unit"),
        }
    }
}

/// Represents an AST annotated with metadata such as types and location info.
#[derive(Debug, Clone)]
pub struct Ast {
    /// The variant of the AST.
    pub ast: AstNode,

    /// The location in the file of the AST.
    pub span: Span,
}

/// Represents the various types of ASTs that are valid for type migration.
#[derive(Debug, Clone)]
pub enum AstNode {
    /// A numeric value. This can be either an integer or a float.
    Number(f64),

    /// A boolean value.
    Boolean(bool),

    /// An infix or binary operator.
    Binary {
        /// The operator being applied.
        op: BinaryOp,

        /// The left hand side.
        left: Box<Ast>,

        /// The right hand side.
        right: Box<Ast>,
    },

    /// A unary operator.
    Unary {
        /// The operator being applied.
        op: UnaryOp,

        /// The argument of the operator.
        value: Box<Ast>,
    },

    /// The ternary operator (ie, cond ? then : elsy)
    Ternary {
        /// The condition of the operator.
        cond: Box<Ast>,

        /// The value on true.
        then: Box<Ast>,

        /// The value on false.
        elsy: Box<Ast>,
    },

    /// A declaration (ie, var x = 2).
    Declare {
        /// The variables declared, with their respective optional init values.
        // TODO: patterns
        vars: Vec<(String, Option<Ast>)>,
    },

    /// An identifier (for variables and constants).
    Identifier(String),

    /// An assignment to a previously declared variable.
    Assign {
        /// The variable being assigned to.
        var: String,

        /// The expression to assign.
        expr: Box<Ast>,
    },

    /// A coercion inserted by the type migrator.
    Coercion {
        /// The expression being coerced.
        expr: Box<Ast>,

        /// The type being coerced from. This is here for redundancy and finding unintentional bugs.
        source_type: Type,

        /// The type being coerced to.
        dest_type: Type,
    },

    /// A block of statements.
    Block(Vec<Ast>),

    /// An if statement
    If {
        /// The condition of the if statement.
        cond: Box<Ast>,

        /// The then clause of the if statement.
        then: Box<Ast>,

        /// The else clause of the if statement.
        elsy: Option<Box<Ast>>,
    },

    /// A while loop.
    While {
        /// The condition of the while loop.
        cond: Box<Ast>,

        /// The body of the while loop.
        body: Box<Ast>,
    },
}

impl Display for Ast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ast)
    }
}

impl Display for AstNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AstNode::Number(v) => write!(f, "{}", v),
            AstNode::Boolean(v) => write!(f, "{}", v),
            AstNode::Binary { op, left, right } => write!(f, "({} {} {})", left, op, right),
            AstNode::Unary { op, value } => write!(f, "{}{}", op, value),
            AstNode::Ternary { cond, then, elsy } => write!(f, "({} ? {} : {})", cond, then, elsy),
            AstNode::Coercion {
                expr, dest_type, ..
            } => write!(f, "({} : {})", expr, dest_type),
            AstNode::Declare { vars } => {
                write!(f, "var ")?;
                let mut first = true;
                for (var, init) in vars.iter() {
                    if first {
                        first = false;
                    } else {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", var)?;
                    if let Some(init) = init {
                        write!(f, " = {}", init)?;
                    }
                }
                write!(f, ";")
            }
            AstNode::Identifier(var) => write!(f, "{}", var),

            AstNode::Assign { var, expr } => write!(f, "({} = {})", var, expr),

            AstNode::Block(block) => {
                writeln!(f, "{{")?;
                for stat in block {
                    writeln!(f, "{};", stat)?;
                }
                writeln!(f, "}}")
            }

            AstNode::If {
                cond,
                then,
                elsy: Some(elsy),
            } => write!(f, "if ({}) {}\nelse {}", cond, then, elsy),
            AstNode::If {
                cond,
                then,
                elsy: None,
            } => write!(f, "if ({}) {}", cond, then),

            AstNode::While { cond, body } => write!(f, "while ({}) {}", cond, body),
        }
    }
}

/// Converts a [`swc_ecma_ast`] module body into a list of ASTs with type annotations for type migration.
///
/// # Example
/// ```rs
/// # use swc_ecma_parser::Parser;
/// # use js2ts::parse;
/// # fn testy(parser: Parser) -> Result<Vec<Ast>, ()> {
/// let module = parser.parse_module()?;
/// let asts = parse::parse(module.body);
/// println!("{:?}", asts);
/// # Ok()
/// # }
/// ```
pub fn parse(module: Vec<ModuleItem>) -> Vec<Ast> {
    module
        .into_iter()
        .filter_map(|item| match item {
            ModuleItem::ModuleDecl(_) => None,
            ModuleItem::Stmt(statement) => walk_statement(statement),
        })
        .collect()
}

fn walk_statement(statement: Stmt) -> Option<Ast> {
    match statement {
        Stmt::Block(block) => Some(Ast {
            ast: AstNode::Block(block.stmts.into_iter().filter_map(walk_statement).collect()),
            span: block.span,
        }),

        Stmt::Empty(_) => None,

        Stmt::Debugger(_) => todo!(),
        Stmt::With(_) => todo!(),
        Stmt::Return(_) => todo!(),
        Stmt::Labeled(_) => todo!(),
        Stmt::Break(_) => todo!(),
        Stmt::Continue(_) => todo!(),

        Stmt::If(if_) => Some(Ast {
            ast: AstNode::If {
                cond: Box::new(walk_expression(*if_.test)),
                then: Box::new(walk_statement(*if_.cons)?),
                elsy: if_.alt.and_then(|v| walk_statement(*v)).map(Box::new),
            },
            span: if_.span,
        }),

        Stmt::Switch(_) => todo!(),
        Stmt::Throw(_) => todo!(),
        Stmt::Try(_) => todo!(),

        Stmt::While(while_) => Some(Ast {
            ast: AstNode::While {
                cond: Box::new(walk_expression(*while_.test)),
                body: Box::new(walk_statement(*while_.body)?),
            },
            span: while_.span,
        }),

        Stmt::DoWhile(_) => todo!(),
        Stmt::For(_) => todo!(),
        Stmt::ForIn(_) => todo!(),
        Stmt::ForOf(_) => todo!(),

        Stmt::Decl(Decl::Class(_)) => todo!(),
        Stmt::Decl(Decl::Fn(_)) => todo!(),

        Stmt::Decl(Decl::Var(decl)) => {
            let mut vars = Vec::new();
            for decl in decl.decls {
                match decl.name {
                    Pat::Ident(name) => {
                        vars.push((name.id.to_string(), decl.init.map(|v| walk_expression(*v))));
                    }

                    _ => todo!(),
                }
            }

            Some(Ast {
                ast: AstNode::Declare { vars },
                span: decl.span,
            })
        }

        Stmt::Decl(Decl::TsInterface(_)) => todo!(),
        Stmt::Decl(Decl::TsTypeAlias(_)) => todo!(),
        Stmt::Decl(Decl::TsEnum(_)) => todo!(),
        Stmt::Decl(Decl::TsModule(_)) => todo!(),

        Stmt::Expr(e) => Some(walk_expression(*e.expr)),
    }
}

fn walk_expression(expression: Expr) -> Ast {
    match expression {
        Expr::This(_) => todo!(),
        Expr::Array(_) => todo!(),
        Expr::Object(_) => todo!(),
        Expr::Fn(_) => todo!(),

        Expr::Unary(unary) => {
            let value = walk_expression(*unary.arg);
            Ast {
                ast: AstNode::Unary {
                    op: unary.op,
                    value: Box::new(value),
                },
                span: unary.span,
            }
        }

        Expr::Update(_) => todo!(),

        Expr::Bin(bin) => {
            let left = walk_expression(*bin.left);
            let right = walk_expression(*bin.right);
            Ast {
                ast: AstNode::Binary {
                    op: bin.op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span: bin.span,
            }
        }

        Expr::Assign(assign) => match assign.left {
            PatOrExpr::Pat(var) => match *var {
                Pat::Ident(var) => Ast {
                    ast: AstNode::Assign {
                        var: var.id.to_string(),
                        expr: Box::new(walk_expression(*assign.right)),
                    },
                    span: assign.span,
                },

                Pat::Array(_) => todo!(),
                Pat::Rest(_) => todo!(),
                Pat::Object(_) => todo!(),
                Pat::Assign(_) => todo!(),
                Pat::Invalid(_) => todo!(),
                Pat::Expr(_) => todo!(),
            },
            PatOrExpr::Expr(_) => todo!(),
        },

        Expr::Member(_) => todo!(),
        Expr::SuperProp(_) => todo!(),

        Expr::Cond(trinary) => {
            let cond = walk_expression(*trinary.test);
            let then = walk_expression(*trinary.cons);
            let elsy = walk_expression(*trinary.alt);
            Ast {
                ast: AstNode::Ternary {
                    cond: Box::new(cond),
                    then: Box::new(then),
                    elsy: Box::new(elsy),
                },
                span: trinary.span,
            }
        }

        Expr::Call(_) => todo!(),
        Expr::New(_) => todo!(),
        Expr::Seq(_) => todo!(),
        Expr::Ident(var) => Ast {
            ast: AstNode::Identifier(var.to_string()),
            span: var.span,
        },

        Expr::Lit(lit) => match lit {
            Lit::Str(_) => todo!(),

            Lit::Bool(b) => Ast {
                ast: AstNode::Boolean(b.value),
                span: b.span,
            },

            Lit::Null(_) => todo!(),

            Lit::Num(n) => Ast {
                ast: AstNode::Number(n.value),
                span: n.span,
            },

            Lit::BigInt(_) => todo!(),
            Lit::Regex(_) => todo!(),
            Lit::JSXText(_) => todo!(),
        },

        Expr::Tpl(_) => todo!(),
        Expr::TaggedTpl(_) => todo!(),
        Expr::Arrow(_) => todo!(),
        Expr::Class(_) => todo!(),
        Expr::Yield(_) => todo!(),
        Expr::MetaProp(_) => todo!(),
        Expr::Await(_) => todo!(),

        Expr::Paren(paren) => walk_expression(*paren.expr),

        Expr::JSXMember(_) => todo!(),
        Expr::JSXNamespacedName(_) => todo!(),
        Expr::JSXEmpty(_) => todo!(),
        Expr::JSXElement(_) => todo!(),
        Expr::JSXFragment(_) => todo!(),
        Expr::TsTypeAssertion(_) => todo!(),
        Expr::TsConstAssertion(_) => todo!(),
        Expr::TsNonNull(_) => todo!(),
        Expr::TsAs(_) => todo!(),
        Expr::TsInstantiation(_) => todo!(),
        Expr::TsSatisfies(_) => todo!(),
        Expr::PrivateName(_) => todo!(),
        Expr::OptChain(_) => todo!(),
        Expr::Invalid(_) => todo!(),
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::*;

    #[test]
    fn empty() {
        parse_helper("");
    }

    #[test]
    fn number() {
        parse_helper("2");
        parse_helper("3.5");
        parse_helper("6e9");
        parse_helper("4e-20");
    }

    #[test]
    fn boolean() {
        parse_helper("true");
        parse_helper("false");
    }

    #[test]
    fn operators() {
        parse_helper("1 + 2");
        parse_helper("1 - 2");
        parse_helper("1 * 2");
        parse_helper("1 / 2");
        parse_helper("1 << 2");
        parse_helper("1 >> 2");
        parse_helper("1 < 2");
        parse_helper("1 > 2");
        parse_helper("1 == 2");
        parse_helper("1 === 2");
        parse_helper("1 != 2");
        parse_helper("1 !== 2");
        parse_helper("1 <= 2");
        parse_helper("1 >= 2");
        parse_helper("1 & 2");
        parse_helper("1 | 2");
        parse_helper("1 ^ 2");
        parse_helper("1 + 2 * 3 / 4");
        parse_helper("-2");
        parse_helper("+true");
        parse_helper("(true + 2) / false");
    }

    #[test]
    fn ternary() {
        parse_helper("true ? 42 : false");
        parse_helper("false ? 2 + 3 : 9 / 5");
        parse_helper("true && false ? 4 - 3 : false || true");
    }

    #[test]
    fn variables() {
        parse_helper("x");
        parse_helper("y");
        parse_helper("x = 3");
        parse_helper("var x = 3");
        parse_helper("var x, y = 4, z, w = 6");
    }

    #[test]
    fn if_() {
        parse_helper("if (true) 2");
        parse_helper("if (true) 2\nelse 3");
        parse_helper("if (true) 2\nelse 3");
        parse_helper("if (true) {2; 3}");
        parse_helper("if (true) 2\nelse {3; 4}");
        parse_helper("if (true) {2; 3} else {3; 5}");
    }

    #[test]
    fn while_() {
        parse_helper("while (true) 2");
        parse_helper("while (true) {2; 3}");
    }
}
