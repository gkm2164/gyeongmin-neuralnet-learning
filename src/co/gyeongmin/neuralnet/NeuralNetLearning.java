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
    private NeuralNetLayer[] hiddenLayers;
    private NeuralNetLayer outputLayer;

    public NeuralNetLearning(double learningRate, double momentumTerm, int[] hiddenLayerNodes, int numInput, int numOutput) {
        this.learningRate = learningRate;
        this.momentumTerm = momentumTerm;

        inputLayer = new NeuralNetLayer(0, numInput);

        hiddenLayers = new NeuralNetLayer[hiddenLayerNodes.length];

        for (int i = 0; i < hiddenLayerNodes.length; i++) {
            hiddenLayers[i] = new NeuralNetLayer(i + 1, hiddenLayerNodes[i]);
        }

        outputLayer = new NeuralNetLayer(hiddenLayerNodes.length, numOutput);

        hiddenLayers[0].setWeightCount(inputLayer.getNodes().length);

        for (int i = 1; i < hiddenLayers.length; i++) {
            hiddenLayers[i].setWeightCount(hiddenLayers[i - 1].getNodes().length);
        }

        outputLayer.setWeightCount(hiddenLayers[hiddenLayerNodes.length - 1].getNodes().length);
    }

    public void forward(double[] values) {
        SigmoidFunction function = x -> 1 / (1 + Math.exp(-x));

        inputLayer.forward(values);
        hiddenLayers[0].forward(inputLayer, function);
        for (int i = 1; i < hiddenLayers.length; i++) {
            hiddenLayers[i].forward(hiddenLayers[i - 1], function);
        }
        outputLayer.forward(hiddenLayers[hiddenLayers.length - 1], function);
    }

    public void backPropagate(double[] answer) {
        int numHiddenLayers = hiddenLayers.length;

        outputLayer.backPropagate(answer, hiddenLayers[numHiddenLayers - 1], learningRate, momentumTerm);

        hiddenLayers[numHiddenLayers - 1].backPropagate(outputLayer, hiddenLayers[numHiddenLayers - 2], learningRate, momentumTerm);
        for (int i = numHiddenLayers - 2; i > 0; i--) {
            hiddenLayers[i].backPropagate(hiddenLayers[i + 1], hiddenLayers[i - 1], learningRate, momentumTerm);
        }
    }

    public double[] getOutput() {
        double[] ret = new double[outputLayer.getNodes().length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = outputLayer.getNodes()[i].getNodeValue();
        }

        return ret;
    }
}
