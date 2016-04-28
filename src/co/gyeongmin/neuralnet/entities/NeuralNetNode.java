package co.gyeongmin.neuralnet.entities;

import java.lang.reflect.Array;

/**
 * Created by USER on 2016-04-28.
 */
public class NeuralNetNode {
    private final int nodeID;
    private double nodeValue;
    private double nodeError;
    private double[] weight;
    private double[] deltaWeight;

    public NeuralNetNode(int nodeID) {
        this.nodeID = nodeID;
    }

    public void setWeightCount(int weightCount) {
        weight = new double[weightCount];
        deltaWeight = new double[weightCount];

        Array.set(weight, 0, 1.0);
        Array.set(deltaWeight, 0, 0.0);
    }

    public void forward(NeuralNetLayer previousLayer, SigmoidFunction sigmoidFunction) {
        NeuralNetNode[] nodes = previousLayer.getNodes();

        double sum = 0.0;

        for (int i = 0; i < nodes.length; i++) {
            sum += nodes[i].nodeValue * weight[i];
        }

        this.nodeValue = sigmoidFunction.calculate(sum);
    }

    public void backPropagate(double desiredValue,
                              NeuralNetLayer previousLayer, double learningRate, double momentumTerm) {
        nodeError = nodeValue * (1 - nodeValue) * (desiredValue - nodeValue);
        updateWeight(previousLayer, learningRate, momentumTerm);
    }

    public void backPropagate(NeuralNetLayer nextLayer,
                              NeuralNetLayer previousLayer, double learningRate, double momentumTerm) {
        NeuralNetNode[] nodes = nextLayer.getNodes();

        this.nodeError = 0.0;

        for (int i = 0; i < nodes.length; i++) {
            this.nodeError = nodes[i].nodeError * nodes[i].weight[nodeID];
        }

        updateWeight(previousLayer, learningRate, momentumTerm);
    }

    public void updateWeight(NeuralNetLayer previousLayer, double learningRate, double momentumTerm) {
        NeuralNetNode[] nodes = previousLayer.nodes;

        for (int i = 0; i < nodes.length; i++) {
            double oldDelta = deltaWeight[i];
            deltaWeight[i] = learningRate * nodeError * nodes[i].nodeValue;
            weight[i] = weight[i] + deltaWeight[i] + (momentumTerm * oldDelta);
        }
    }

    public void setNodeValue(double nodeValue) {
        this.nodeValue = nodeValue;
    }

    public double getNodeValue() {
        return nodeValue;
    }
}
