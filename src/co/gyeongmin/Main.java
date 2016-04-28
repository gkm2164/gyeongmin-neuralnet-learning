package co.gyeongmin;

import co.gyeongmin.neuralnet.NeuralNetLearning;

public class Main {

    public static void main(String[] args) {
        double[] inputs = { 0.0, 1.0, 5.0, 9.0, 10.0 };
        double[] answer = { 1.0, 2.0, 3.0, 4.0, 5.0 };

        NeuralNetLearning neuralNetLearning = new NeuralNetLearning(0.45, 0.9, new int[]{ 11, 11, 11, 11, 11 }, 1, 1);

            int iterate = 1000;
            while (iterate-- > 0) {
                for (int i = 0; i < 5; i++) {
                    neuralNetLearning.forward(new double[]{convertIntoPoint(inputs[i], 0, 10)});
                    neuralNetLearning.backPropagate(new double[]{pointToValue(answer[i], 1, 5)});
                }
            }

        for (int i = 0; i < 11; i++) {
            neuralNetLearning.forward(new double[]{ convertIntoPoint(i, 0, 10) });
            double[] retVal = neuralNetLearning.getOutput();
            System.out.println("(" + i + ")" + "Return: " + pointToValue(retVal[0], 1, 5));
        }
    }

    public static double convertIntoPoint(double value, double min, double max) {
        double val = value - min;
        double range = max - min;

        return val / range;
    }

    public static double pointToValue(double ptr, double min, double max) {
        double range = max - min;
        double val = ptr * range;
        return val + min;
    }
}
