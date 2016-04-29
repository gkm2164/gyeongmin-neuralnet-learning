package co.gyeongmin.neuralnet.entities;

/**
 * Created by USER on 2016-04-28.
 */
public class NeuralNetLayer {
    private final NeuralNetNode[] nodes;
    private NeuralNetLayer prevLayer = null;
    private NeuralNetLayer nextLayer = null;

    public NeuralNetLayer(int numNodes) {
        this.nodes = new NeuralNetNode[numNodes];

        for (int i = 0; i < numNodes; i++) {
            nodes[i] = new NeuralNetNode(i);
        }

        LayerName.addLayer(this);
    }

    public void connectPreviousLayer(NeuralNetLayer previousLayer) {
        this.prevLayer = previousLayer;
        prevLayer.setNextLayer(this);

        int numPreviousLayerNode = previousLayer.nodes.length;

        for (NeuralNetNode node : nodes) {
            node.setWeightCount(numPreviousLayerNode);
        }
    }

    public NeuralNetNode[] getNodes() {
        return nodes;
    }

    public void forward(double[] values,
                        SigmoidFunction sigmoidFunction) {
        for (int i = 0; i < nodes.length; i++) {
            nodes[i].setNodeValue(values[i]);
        }

        nextLayer.forward(sigmoidFunction);
    }

    public void backPropagate(double[] desiredValue,
                              double learningRate, double momentumTerm) {
        for (int i = 0; i < nodes.length; i++) {
            nodes[i].backPropagate(desiredValue[i], prevLayer, learningRate, momentumTerm);
        }

        prevLayer.backPropagate(learningRate, momentumTerm);
    }

    private void forward(SigmoidFunction sigmoidFunction) {
        for (NeuralNetNode node : nodes) {
            node.forward(prevLayer, sigmoidFunction);
        }

        if (nextLayer != null) {
            nextLayer.forward(sigmoidFunction);
        }
    }

    private void backPropagate(double learningRate, double momentumTerm) {
        if (prevLayer == null) return;

        for (NeuralNetNode node : nodes) {
            node.backPropagate(nextLayer, prevLayer,
                               learningRate, momentumTerm);
        }

        prevLayer.backPropagate(learningRate, momentumTerm);
    }

    public void setNextLayer(NeuralNetLayer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public void showNetStat() {
        for (int i = 0; i < nodes.length; i++) {
            nodes[i].showWeights("Layer #" + LayerName.nnlHashMap.get(this) + ", " + i + " :: ");
        }

        if (nextLayer != null) {
            nextLayer.showNetStat();
        }
    }
}
