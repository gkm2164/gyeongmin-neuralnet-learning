package co.gyeongmin.nntest;

import co.gyeongmin.neuralnet.NeuralNetLearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by gyeongmin on 4/29/16.
 */
public class OpticalRecognitionFunction {

    private final double VALUE_MAX = 9.0;
    private final double VALUE_MIN = 0.0;

    private String filename;
    private NeuralNetLearning neuralNetLearning;
    private List<TestSet> testSetList;
    public OpticalRecognitionFunction(String filename) throws IOException {
        this.filename = filename;
        this.neuralNetLearning = new NeuralNetLearning(0.45, 0.9, new int[]{128, 64, 32, 16, 8, 4, 2}, 256, 1);
        this.testSetList = new ArrayList<>();

        study();
    }

    private void readTestSetList() throws IOException {
        int iterate = 10;
        BufferedReader br = new BufferedReader(new FileReader(filename));

        for (String temp = br.readLine(); iterate-- > 0 && temp != null; temp = br.readLine()) {
            String[] lines = new String[32];

            lines[0] = temp;
            for (int i = 1; i < 32; i++) {
                lines[i] = br.readLine();
            }

            temp = br.readLine();
            temp = temp.replaceAll(" ", "");
            int value = Integer.valueOf(temp);

            TestSet ts = new TestSet(lines, value);
            testSetList.add(ts);
        }
    }

    private void study() throws IOException {
        readTestSetList();

        double range = VALUE_MAX - VALUE_MIN;

        for (int i = 0; i < 10000; i++) {
            testSetList.forEach(ts -> {
                neuralNetLearning.forward(ts.data);
                neuralNetLearning.backPropagate(new double[]{(ts.answer - VALUE_MIN) / range});
            });

            if ((i + 1) % 2000 == 0) {
                System.out.println("#" + (i + 1) + " times studied.");
            }
        }
    }

    private double areaAverage(byte[][] data, int startX, int startY, int lenX, int lenY) {
        int size = lenX * lenY;
        double sum = 0.0;
        for (int i = startX; i < startX + lenX; i++) {
            for (int j = startY; j < startY + lenY; j++) {
                sum += data[i][j] - '0';
            }
        }

        return Math.round(sum / size);
    }

    public int number(byte[][] data) {

        double[] dblData = new double[16 * 16];

        for (int i = 0; i < data.length; i += 2) {
            for (int j = 0; j < data[i].length; j += 2) {
                int iIdx = i / 2;
                int jIdx = j / 2;
                double value = areaAverage(data, i, j, 2, 2);
                dblData[iIdx * 16 + jIdx] = value;
            }
        }

        neuralNetLearning.forward(dblData);
        return (int)Math.round(neuralNetLearning.getOutput()[0] * (VALUE_MAX - VALUE_MIN) + VALUE_MIN);
    }

    private class TestSet {
        private double[] data;
        private int answer;

        TestSet(String[] lines, int value) {

            data = new double[16 * 16];
            saveData(lines);
            this.answer = value;
        }

        private void saveData(String[] lines) {
            byte[][] tempData = new byte[lines.length][];

            for (int i = 0; i < tempData.length; i++) {
                tempData[i] = lines[i].getBytes();
            }

            for (int i = 0; i < lines.length; i += 2) {
                for (int j = 0; j < 32; j += 2) {
                    int iidx = i / 2,
                        jidx = j / 2;
                    data[iidx * 16 + jidx] = areaAverage(tempData, i, j, 2, 2);
                }
            }
        }
    }
}
