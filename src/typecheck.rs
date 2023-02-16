use crate::parse::{Ast, AstNode, Type};

fn typecheck_helper(ast: &Ast) -> Result<Type, String> {
    match &ast.ast {
        AstNode::Number(_) => Ok(Type::Number),
        AstNode::Boolean(_) => Ok(Type::Bool),
        AstNode::Binary { .. } => todo!(),
        AstNode::Unary { .. } => todo!(),

        AstNode::Ternary { cond, then, elsy } => {
            if typecheck_helper(&**cond)? != Type::Bool {
                return Err("condition of ternary operator must be of type `bool`".to_string());
            }

            let t1 = typecheck_helper(&**then)?;

            if t1 != typecheck_helper(&**elsy)? {
                Err("branches of ternary operator must be of the same type".to_string())
            } else {
                Ok(t1)
            }
        }

        AstNode::Coercion { expr, source_type, dest_type } => {
            if typecheck_helper(&**expr)? != *source_type {
                Err("coercion source type must match expression type".to_string())
            } else {
                Ok(dest_type.clone())
            }
        }
    }
}

/// Performs type checking on a migrated [`Ast`] to make sure the resulting Ast is valid.
pub fn typecheck(asts: &[Ast]) -> Result<(), String> {
    for ast in asts {
        typecheck_helper(ast)?;
    }

    Ok(())
}