package co.gyeongmin.nntest;

import co.gyeongmin.neuralnet.NeuralNetLearning;

/**
 * Created by USER on 2016-04-29.
 */
public class XORFunction {
    private double[][] inputs = {
            { 0., 0. },
            { 0., 1. },
            { 1., 0. },
            { 1., 1. }
    };

    private double[][] answers = {
            { 0. },
            { 1. },
            { 1. },
            { 0. }
    };

    private final int ITERATE = 10000;

    private NeuralNetLearning neuralNetLearning;
    public XORFunction() {
        neuralNetLearning = new NeuralNetLearning(0.45, 0.9, new int[]{2, 2}, 2, 1);

        study();
    }

    public void study() {
        int c = ITERATE;

        while (c-- > 0) {
            for (int i = 0; i < inputs.length; i++) {
                neuralNetLearning.forward(inputs[i]);
                neuralNetLearning.backPropagate(answers[i]);
            }
        }
    }

    public int xor(int x1, int x2) {
        double[] input = { x1, x2 };

        neuralNetLearning.forward(input);
        double[] answer = neuralNetLearning.getOutput();

        return (int)Math.round(answer[0]);
    }
}
