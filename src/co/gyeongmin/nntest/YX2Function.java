package co.gyeongmin.nntest;

import co.gyeongmin.neuralnet.NeuralNetLearning;

/**
 * Created by USER on 2016-04-29.
 */
public class YX2Function {
    private final int ITERATE = 10000;

    private double min;
    private double max;

    private NeuralNetLearning neuralNetLearning;

    public YX2Function(double max) {
        this.min = -max;
        this.max = max;

        neuralNetLearning = new NeuralNetLearning(0.45, 0.9, new int[]{5, 5, 5, 5, 5}, 1, 1);
        study();
    }

    public void study() {
        double[] tempX = new double[1],
                 tempY = new double[1];
        int c = ITERATE;

        while (c-- > 0) {
            double dx = (max - min) / 50;
            for (int i = 0; i < 50; i++) {
                double value = dx * i + min;
                double ret = value * value;
                double scaledX = (value - min) / (max - min),
                       scaledY = (ret - 0) / (Math.pow(max, 2) - 0);

                tempX[0] = scaledX;
                tempY[0] = scaledY;
                neuralNetLearning.forward(tempX);
                neuralNetLearning.backPropagate(tempY);
            }
        }
    }

    public double run(double x) {
        double input = (x - min) / (max - min);

        neuralNetLearning.forward(new double[] { input });
        double ret = neuralNetLearning.getOutput()[0];

        return ret * Math.pow(max, 2);
    }
}
