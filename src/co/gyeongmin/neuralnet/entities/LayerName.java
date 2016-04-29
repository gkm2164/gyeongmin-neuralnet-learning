package co.gyeongmin.neuralnet.entities;

import java.util.HashMap;

/**
 * Created by USER on 2016-04-29.
 */
public class LayerName {
    public static Object obj;
    public static Integer id = 0;
    public static HashMap<NeuralNetLayer, Integer> nnlHashMap = new HashMap<>();

    public static void addLayer(NeuralNetLayer nnl) {
        id++;

        nnlHashMap.put(nnl, id);
    }
}
