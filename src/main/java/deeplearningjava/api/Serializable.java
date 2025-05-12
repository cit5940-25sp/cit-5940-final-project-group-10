package deeplearningjava.api;

import java.io.IOException;

/**
 * Interface for models that can be saved to and loaded from files.
 */
public interface Serializable {
    
    /**
     * Saves the model to a file.
     * @param filePath Path to save the model to
     * @throws IOException If an I/O error occurs
     */
    void save(String filePath) throws IOException;
    
    /**
     * Loads the model from a file.
     * @param filePath Path to load the model from
     * @throws IOException If an I/O error occurs
     * @return The loaded model
     */
    Serializable load(String filePath) throws IOException;
}