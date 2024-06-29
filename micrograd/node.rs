use std::cmp::Ordering;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Sub;

type Id = &'static str;

#[derive(Debug, Clone)]
enum Operation {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone)]
enum Node {
    Weight(Id, f64),
    Bias(Id, f64),
    Input(Id, f64),
    Intermediate(Operation, f64),
}

impl Node {
    pub fn value(&self) -> f64 {
        match self {
            Node::Weight(_, v)
            | Node::Bias(_, v)
            | Node::Input(_, v)
            | Node::Intermediate(_, v) => *v,
        }
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl Eq for Node {}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use Node::*;
        match (self, other) {
            (Input(_, a), Input(_, b))
            | (Weight(_, a), Weight(_, b))
            | (Bias(_, a), Bias(_, b))
            | (Intermediate(_, a), Intermediate(_, b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl Add for Node {
    type Output = Node;

    fn add(self, other: Node) -> Node {
        Node::Intermediate(Operation::Add, self.value() + other.value())
    }
}

impl Sub for Node {
    type Output = Node;

    fn sub(self, other: Node) -> Node {
        Node::Intermediate(Operation::Sub, self.value() - other.value())
    }
}

impl Div for Node {
    type Output = Node;

    fn div(self, other: Node) -> Node {
        Node::Intermediate(Operation::Div, self.value().div(other.value()))
    }
}

impl Mul for Node {
    type Output = Node;

    fn mul(self, other: Node) -> Node {
        Node::Intermediate(Operation::Mul, self.value() * other.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node() {
        let input_node = Node::Input("1", 3.0);
        let weight_node = Node::Weight("2", 7.0);
        let bias_node = Node::Bias("3", 1.0);

        let result = (input_node * weight_node) + bias_node;
        assert_eq!(result, Node::Intermediate(Operation::Add, 22.0));
    }
}
