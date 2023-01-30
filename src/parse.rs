use swc_common::Span;
use swc_ecma_ast::{ModuleItem, Stmt, Expr, Lit, BinaryOp, UnaryOp};

/// Represents a type.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Type {
    /// An unknown type.
    #[default]
    Unknown,

    /// A number type.
    Number,

    /// A boolean type.
    Bool,
}

/// Represents an AST annotated with metadata such as types and location info.
#[derive(Debug)]
pub struct Ast {
    /// The variant of the AST.
    pub ast: AstVariants,

    /// The location in the file of the AST.
    pub span: Span,

    /// The type of the ast.
    pub type_: Type,
}

/// Represents the various types of ASTs that are valid for type migration.
#[derive(Debug)]
pub enum AstVariants {
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
    module.into_iter().filter_map(|item| {
        match item {
            ModuleItem::ModuleDecl(_) => None,
            ModuleItem::Stmt(statement) => walk_statement(statement),
        }
    }).collect()
}

fn walk_statement(statement: Stmt) -> Option<Ast> {
    match statement {
        Stmt::Block(_) => todo!(),

        Stmt::Empty(_) => None,

        Stmt::Debugger(_) => todo!(),
        Stmt::With(_) => todo!(),
        Stmt::Return(_) => todo!(),
        Stmt::Labeled(_) => todo!(),
        Stmt::Break(_) => todo!(),
        Stmt::Continue(_) => todo!(),
        Stmt::If(_) => todo!(),
        Stmt::Switch(_) => todo!(),
        Stmt::Throw(_) => todo!(),
        Stmt::Try(_) => todo!(),
        Stmt::While(_) => todo!(),
        Stmt::DoWhile(_) => todo!(),
        Stmt::For(_) => todo!(),
        Stmt::ForIn(_) => todo!(),
        Stmt::ForOf(_) => todo!(),
        Stmt::Decl(_) => todo!(),

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
                ast: AstVariants::Unary {
                    op: unary.op,
                    value: Box::new(value),
                },
                span: unary.span,
                type_: Default::default(),
            }
        }

        Expr::Update(_) => todo!(),

        Expr::Bin(bin) => {
            let left = walk_expression(*bin.left);
            let right = walk_expression(*bin.right);
            Ast {
                ast: AstVariants::Binary {
                    op: bin.op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span: bin.span,
                type_: Default::default(),
            }
        }

        Expr::Assign(_) => todo!(),
        Expr::Member(_) => todo!(),
        Expr::SuperProp(_) => todo!(),
        Expr::Cond(_) => todo!(),
        Expr::Call(_) => todo!(),
        Expr::New(_) => todo!(),
        Expr::Seq(_) => todo!(),
        Expr::Ident(_) => todo!(),

        Expr::Lit(lit) => {
            match lit {
                Lit::Str(_) => todo!(),

                Lit::Bool(b) => Ast {
                    ast: AstVariants::Boolean(b.value),
                    span: b.span,
                    type_: Type::Bool,
                },

                Lit::Null(_) => todo!(),

                Lit::Num(n) => Ast {
                    ast: AstVariants::Number(n.value),
                    span: n.span,
                    type_: Type::Number,
                },

                Lit::BigInt(_) => todo!(),
                Lit::Regex(_) => todo!(),
                Lit::JSXText(_) => todo!(),
            }
        }

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
