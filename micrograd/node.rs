use std::cell::RefCell;
use std::ops::Add;
use std::ops::Mul;
use std::rc::Rc;

#[derive(Debug, Clone, Copy)]
pub enum Operand {
    Add,
    Sub,
    Mul,
    Leaf,
}

#[derive(Debug)]
pub struct GraphNode {
    raw: f64,
    grad: RefCell<f64>,
    operand: Operand,
    left: Option<Rc<GraphNode>>,
    right: Option<Rc<GraphNode>>,
}

impl From<f64> for GraphNode {
    fn from(value: f64) -> Self {
        Self {
            raw: value,
            operand: Operand::Leaf,
            left: None,
            right: None,
            grad: RefCell::new(0.0),
        }
    }
}

impl GraphNode {
    pub fn new(
        value: f64,
        operand: Operand,
        left: Option<Rc<GraphNode>>,
        right: Option<Rc<GraphNode>>,
    ) -> Self {
        Self {
            raw: value,
            operand,
            left,
            right,
            grad: RefCell::new(0.0),
        }
    }

    fn backprop(&self) {
        match self.operand {
            Operand::Add | Operand::Sub => {
                let own_grad = self.grad.borrow_mut();

                if let Some(left) = &self.left {
                    let mut left_grad = left.grad.borrow_mut();
                    *left_grad += *own_grad;
                }

                if let Some(right) = &self.right {
                    let mut right_grad = right.grad.borrow_mut();
                    *right_grad += *own_grad;
                }
            }
            Operand::Mul => {
                let own_grad = self.grad.borrow_mut();

                if let (Some(left), Some(right)) = (&self.left, &self.right) {
                    let mut left_grad = left.grad.borrow_mut();
                    let mut right_grad = right.grad.borrow_mut();
                    // Chain rule: differentiate over l * r -> dl/dl * dr/dl + dl/dr * dr/dr
                    // dl/dl * dr/dl -> 1 * dr/dl
                    let dleft = (*right_grad).mul(*own_grad);
                    *left_grad += dleft;

                    // dl/dr * dr/dr -> 1 * dl/dr
                    let dright = (*left_grad).mul(*own_grad);
                    *right_grad += dright;
                }
            }
            _ => {}
        };
    }
}

impl PartialEq for GraphNode {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

#[derive(Debug, PartialEq)]
pub enum Node {
    Intermediate(Rc<GraphNode>),
    Input(Rc<GraphNode>),
    Weight(Rc<GraphNode>),
    Bias(Rc<GraphNode>),
}

pub trait Value {
    fn value(&self) -> f64;
}

impl Node {
    fn inner(&self) -> Rc<GraphNode> {
        match self {
            Node::Intermediate(g) => Rc::clone(g),
            Node::Input(g) | Node::Weight(g) | Node::Bias(g) => Rc::clone(g),
        }
    }
}

impl Value for Node {
    fn value(&self) -> f64 {
        self.inner().raw
    }
}

impl Add for Node {
    type Output = Node;

    fn add(self, other: Self) -> Self::Output {
        Node::Intermediate(Rc::new(GraphNode {
            raw: self.value() + other.value(),
            operand: Operand::Add,
            left: Some(self.inner()),
            right: Some(other.inner()),
            grad: RefCell::new(0.0),
        }))
    }
}

impl Mul for Node {
    type Output = Node;

    fn mul(self, other: Self) -> Self::Output {
        Node::Intermediate(Rc::new(GraphNode {
            raw: self.value() * other.value(),
            operand: Operand::Mul,
            left: Some(self.inner()),
            right: Some(other.inner()),
            grad: RefCell::new(0.0),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_sanity_check() {
        let x_gn: f64 = 10.0;
        let x_gn: GraphNode = x_gn.into();
        let x = Node::Input(x_gn.into());

        let w_gn: f64 = 3.0;
        let w_gn: GraphNode = w_gn.into();
        let w = Node::Weight(w_gn.into());

        let b_gn: f64 = 7.0;
        let b_gn: GraphNode = b_gn.into();
        let b = Node::Bias(b_gn.into());

        let g = (x * w) + b;
        assert_eq!(37.0, g.value());
    }
}
