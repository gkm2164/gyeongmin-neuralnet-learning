package co.gyeongmin.neuralnet;

import co.gyeongmin.neuralnet.entities.NeuralNetLayer;
import co.gyeongmin.neuralnet.entities.SigmoidFunction;

import java.util.LinkedList;

/**
 * Created by USER on 2016-04-28.
 */
public class NeuralNetLearning {

    private final double learningRate;
    private final double momentumTerm;

    private NeuralNetLayer inputLayer;
    private NeuralNetLayer outputLayer;

    public NeuralNetLearning(double learningRate, double momentumTerm,
                             int[] hiddenLayerNodes, int numInput, int numOutput) {
        this.learningRate = learningRate;
        this.momentumTerm = momentumTerm;

        inputLayer = new NeuralNetLayer(numInput);

        NeuralNetLayer[] hiddenLayers = new NeuralNetLayer[hiddenLayerNodes.length];
        for (int i = 0; i < hiddenLayerNodes.length; i++) {
            hiddenLayers[i] = new NeuralNetLayer(hiddenLayerNodes[i]);
        }

        outputLayer = new NeuralNetLayer(numOutput);

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
}
