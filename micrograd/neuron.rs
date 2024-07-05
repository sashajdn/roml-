use super::node::{GraphNode, Node, Operand, Value};
use rand::Rng;

struct Neuron<const DIMENSION: usize> {
    w: Box<[Node; DIMENSION]>,
    b: Box<[Node; DIMENSION]>,
}

impl<const DIMENSION: usize> Neuron<DIMENSION> {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();

        let w: Vec<Node> = (0..DIMENSION)
            .map(|_| {
                let r: f64 = rng.gen();
                let gn: GraphNode = r.into();
                Node::Weight(gn.into())
            })
            .collect();

        // TODO: since this can panic; we need to handle this case.
        let w: [Node; DIMENSION] = w.try_into().unwrap();

        let b: Vec<Node> = (0..DIMENSION)
            .map(|_| {
                let r: f64 = rng.gen();
                let gn: GraphNode = r.into();
                Node::Weight(gn.into())
            })
            .collect();

        // TODO: since this can panic; we need to handle this case.
        let b: [Node; DIMENSION] = b.try_into().unwrap();

        Self {
            w: Box::new(w),
            b: Box::new(b),
        }
    }

    pub fn forward_step(&self, input: &[Node; DIMENSION]) -> Node {
        let activation = input.iter().zip(self.w.iter()).zip(self.b.iter()).fold(
            Node::Intermediate(GraphNode::new(0.0, Operand::Add, None, None).into()),
            |acc, ((x, w), b)| {
                let x = (*x).value();
                let w = w.value();
                let b = b.value();
                let res = (x * w) + b;
                let res: GraphNode = res.into();
                acc + Node::Intermediate(res.into())
            },
        );

        // TODO: Wrap with some non-linear function such as ReLu.
        activation
    }
}
