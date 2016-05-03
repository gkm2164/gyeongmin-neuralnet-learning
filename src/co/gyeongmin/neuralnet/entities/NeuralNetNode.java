package co.gyeongmin.neuralnet.entities;

import java.lang.reflect.Array;
import java.time.Instant;
import java.util.Random;
import java.util.stream.DoubleStream;

/**
 * Created by USER on 2016-04-28.
 */
public class NeuralNetNode {
    private final int nodeID;
    private double nodeValue;
    private double nodeError;
    private double[] weight;
    private double[] deltaWeight;

    NeuralNetNode(int nodeID) {
        this.nodeID = nodeID;
    }

    void setWeightCount(int weightCount, double weightMin, double weightMax) {
        weight = new double[weightCount];
        deltaWeight = new double[weightCount];

        double range = weightMax - weightMin;

        for (int i = 0; i < weight.length; i++) {
            double value = Math.random() * range;
            weight[i] = value + weightMin;
        }
        Array.set(deltaWeight, 0, 0.0);
    }

    void forward(NeuralNetLayer previousLayer, SigmoidFunction sigmoidFunction) {
        NeuralNetNode[] nodes = previousLayer.getNodes();

        double sum = 0.0;

        for (int i = 0; i < nodes.length; i++) {
            sum += nodes[i].nodeValue * weight[i];
        }

        this.nodeValue = sigmoidFunction.calculate(sum);
    }

    void backPropagate(double desiredValue, NeuralNetLayer previousLayer,
                       double learningRate, double momentumTerm) {
        nodeError = nodeValue * (1 - nodeValue) * (desiredValue - nodeValue);
        updateWeight(previousLayer, learningRate, momentumTerm);
    }

    void backPropagate(NeuralNetLayer nextLayer, NeuralNetLayer previousLayer,
                       double learningRate, double momentumTerm) {
        NeuralNetNode[] nodes = nextLayer.getNodes();

        double sum = 0.0;
        for (NeuralNetNode node : nodes) {
            sum += node.nodeError * node.weight[nodeID];
        }

        nodeError = nodeValue * (1 - nodeValue) * sum;
        updateWeight(previousLayer, learningRate, momentumTerm);
    }

    private void updateWeight(NeuralNetLayer previousLayer,
                              double learningRate, double momentumTerm) {
        NeuralNetNode[] nodes = previousLayer.getNodes();

        for (int i = 0; i < nodes.length; i++) {
            double oldDelta = deltaWeight[i];
            deltaWeight[i] = learningRate * nodeError * nodes[i].nodeValue;
            weight[i] = weight[i] + deltaWeight[i] + momentumTerm * oldDelta;
        }
    }

    void setNodeValue(double nodeValue) {
        this.nodeValue = nodeValue;
    }

    public double getNodeValue() {
        return nodeValue;
    }

    public void showWeights(String tag) {
        System.out.print(tag + ": ");
        if (weight != null) {
            for (int i = 0; i < weight.length; i++) {
                System.out.format("%.2f, ", weight[i]);
            }
        }
        System.out.println();
    }
}
