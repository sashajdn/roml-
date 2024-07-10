use super::node::{GraphNode, Node, Operand, Value};
use rand::Rng;

struct Neuron<const DIMENSION: usize> {
    w: Box<[Node; DIMENSION]>,
    b: Box<[Node; DIMENSION]>,
}

impl<const DIMENSION: usize> Neuron<DIMENSION> {
    pub fn new(set_random: bool) -> Self {
        let mut rng = rand::thread_rng();

        let w: Vec<Node> = (0..DIMENSION)
            .map(|_| {
                let r: f64 = if set_random { rng.gen() } else { 1.0 };
                let gn: GraphNode = r.into();
                Node::Weight(gn.into())
            })
            .collect();

        // TODO: since this can panic; we need to handle this case.
        let w: [Node; DIMENSION] = w.try_into().expect("BUG: convert vec to array");

        let b: Vec<Node> = (0..DIMENSION)
            .map(|_| {
                let r: f64 = if set_random { rng.gen() } else { 1.0 };
                let gn: GraphNode = r.into();
                Node::Weight(gn.into())
            })
            .collect();

        // TODO: since this can panic; we need to handle this case.
        let b: [Node; DIMENSION] = b.try_into().expect("BUG: convert vec to array");

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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn neuron_sanity_check() {
        let neuron = Neuron::<3>::new(false);

        let node_i = Node::Input(GraphNode::new(10.0, Operand::Add, None, None).into());
        let node_j = Node::Input(GraphNode::new(20.0, Operand::Add, None, None).into());
        let node_k = Node::Input(GraphNode::new(30.0, Operand::Add, None, None).into());

        let input: [Node; 3] = [node_i, node_j, node_k];

        let forward_node = neuron.forward_step(&input);
        assert_eq!(forward_node.value(), 63.0);

        forward_node.inner().backprop();

        let gradient_at_node_i = input[0].grad();
        let gradient_at_node_j = input[1].grad();
        let gradient_at_node_k = input[2].grad();

        assert_ne!(gradient_at_node_i, 0.0);

        println!("Grading at node i: {}", gradient_at_node_i);
        println!("Grading at node j: {}", gradient_at_node_j);
        println!("Grading at node k: {}", gradient_at_node_k);
    }
}
