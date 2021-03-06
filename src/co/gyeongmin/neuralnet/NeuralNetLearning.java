package co.gyeongmin.neuralnet;

import co.gyeongmin.neuralnet.entities.NeuralNetLayer;
import co.gyeongmin.neuralnet.entities.SigmoidFunction;

/**
 * Created by USER on 2016-04-28.
 */
public class NeuralNetLearning {

    private final double learningRate;
    private final double momentumTerm;

    private NeuralNetLayer inputLayer;
    private NeuralNetLayer outputLayer;

    /* Entry functions */
    public NeuralNetLearning(double learningRate, double momentumTerm,
                             int[] hiddenLayerNodes, int numInput, int numOutput) {
        this.learningRate = learningRate;
        this.momentumTerm = momentumTerm;

        /* The layers are connected as doubly linked list. */

        inputLayer = new NeuralNetLayer(numInput);

        NeuralNetLayer[] hiddenLayers = new NeuralNetLayer[hiddenLayerNodes.length];
        for (int i = 0; i < hiddenLayerNodes.length; i++) {
            hiddenLayers[i] = new NeuralNetLayer(hiddenLayerNodes[i]);
        }

        outputLayer = new NeuralNetLayer(numOutput);

        /* As you can see below, input, hiddens, and output are connected */
        NeuralNetLayer nnlPrev = inputLayer;
        for (int i = 0; i < hiddenLayers.length; i++) {
            hiddenLayers[i].connectPreviousLayer(nnlPrev);
            nnlPrev = hiddenLayers[i];
        }
        outputLayer.connectPreviousLayer(hiddenLayers[hiddenLayerNodes.length - 1]);
    }

    public void forward(double[] values) {
        SigmoidFunction function = (x) -> 1 / (1 + Math.exp(-x));
        inputLayer.forward(values, function);
    }

    public void backPropagate(double[] answer) {
        outputLayer.backPropagate(answer, learningRate, momentumTerm);
    }

    public double[] getOutput() {
        double[] ret = new double[outputLayer.getNodes().length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = outputLayer.getNodes()[i].getNodeValue();
        }

        return ret;
    }

    public void showNetStat() {
        inputLayer.showNetStat();
    }
}
