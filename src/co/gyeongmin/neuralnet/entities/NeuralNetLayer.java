package co.gyeongmin.neuralnet.entities;

/**
 * Created by USER on 2016-04-28.
 */
public class NeuralNetLayer {
    final int layerID;
    final NeuralNetNode[] nodes;

    public NeuralNetLayer(int layerID, int numNodes) {
        this.layerID = layerID;
        this.nodes = new NeuralNetNode[numNodes];

        for (int i = 0; i < numNodes; i++) {
            nodes[i] = new NeuralNetNode(i);
        }
    }

    public void connectLayer(NeuralNetLayer previousLayer) {
        int numPreviousLayerNode = previousLayer.nodes.length;
        for (int i = 0; i < nodes.length; i++) {
            nodes[i].setWeightCount(numPreviousLayerNode);
        }
    }

    public NeuralNetNode[] getNodes() {
        return nodes;
    }

    public void forward(NeuralNetLayer previousLayer, SigmoidFunction sigmoidFunction) {
        for (NeuralNetNode node: nodes) {
            node.forward(previousLayer, sigmoidFunction);
        }
    }

    public void setWeightCount(int weightCount) {
        for (int i = 0; i < nodes.length; i++) {
            nodes[i].setWeightCount(weightCount);
        }
    }

    public void forward(double[] values) {
        for (int i = 0; i < nodes.length; i++) {
            nodes[i].setNodeValue(values[i]);
        }
    }

    public void backPropagate(double[] desiredValue,
                              NeuralNetLayer previousLayer, double learningRate, double momentumTerm) {
        for (int i = 0; i < nodes.length; i++) {
            nodes[i].backPropagate(desiredValue[i], previousLayer, learningRate, momentumTerm);
        }
    }

    public void backPropagate(NeuralNetLayer nextLayer,
                              NeuralNetLayer previousLayer, double learningRate, double momentumTerm) {
        for (int i = 0; i < nodes.length; i++) {
            nodes[i].backPropagate(nextLayer, previousLayer, learningRate, momentumTerm);
        }
    }
}
