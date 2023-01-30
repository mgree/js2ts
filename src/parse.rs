use swc_common::Span;
use swc_ecma_ast::{ModuleItem, Stmt, Expr, Lit};

/// Represents a type.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Type {
    /// An unknown type.
    #[default]
    Unknown,

    /// An integer type.
    Int,

    /// A floating point type.
    Float,

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
}

///
/// # Example
/// ```rs
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
        Stmt::Expr(e) => walk_expression(*e.expr),
    }
}

fn walk_expression(expression: Expr) -> Option<Ast> {
    match expression {
        Expr::This(_) => todo!(),
        Expr::Array(_) => todo!(),
        Expr::Object(_) => todo!(),
        Expr::Fn(_) => todo!(),
        Expr::Unary(_) => todo!(),
        Expr::Update(_) => todo!(),
        Expr::Bin(_) => todo!(),
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

                Lit::Bool(b) => Some(Ast {
                    ast: AstVariants::Boolean(b.value),
                    span: b.span,
                    type_: Default::default(),
                }),

                Lit::Null(_) => todo!(),

                Lit::Num(n) => Some(Ast {
                    ast: AstVariants::Number(n.value),
                    span: n.span,
                    type_: Default::default(),
                }),

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
        Expr::Paren(_) => todo!(),
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
