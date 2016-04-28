package co.gyeongmin;

import co.gyeongmin.neuralnet.NeuralNetLearning;

public class Main {

    public static void main(String[] args) {
        double[][] inputs = { { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 } };
        double[] answer = { 0.0, 1.0, 0.0, 1.0 };

        NeuralNetLearning neuralNetLearning = new NeuralNetLearning(0.45, 0.9, new int[]{ 2, 2 }, 2, 1);

            int iterate = 100000;
            while (iterate-- > 0) {
                for (int i = 0; i < 4; i++) {
                    neuralNetLearning.forward(inputs[i]);
                    neuralNetLearning.backPropagate(new double[]{answer[i]});
                }
            }

        for (int i = 0; i < 4; i++) {
            neuralNetLearning.forward(inputs[i]);
            double[] retVal = neuralNetLearning.getOutput();
            System.out.println("(" + inputs[i][0] + ", " + inputs[i][1] + ")" + "Return: " + retVal[0]);
        }
    }
}
