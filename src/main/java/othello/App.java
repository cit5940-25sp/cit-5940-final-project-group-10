package othello;

import javafx.application.Platform;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.scene.control.TextInputDialog;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Optional;

import othello.gamelogic.strategies.StrategyFactory;
import othello.gui.GameController;


/**
 * Controller used to manipulate the GUI of the game.
 */
public class App extends javafx.application.Application {

    // The program arguments MUST match one of these items!
    // Edit this list to add more items!
    private final List<String> acceptedArgs = List.of("human", "minimax", "expectimax", "mcts", "custom", "tensor", "onnx");

    @Override
    public void start(Stage stage) throws IOException {
        Parameters params = getParameters();
        List<String> argList = params.getRaw();
        
        if (argList.size() != 2) {
            System.err.println("Error: Did not provide 2 program arguments");
            System.exit(1);
        }
        
        if (!acceptedArgs.contains(argList.get(0)) || !acceptedArgs.contains(argList.get(1))) {
            System.err.println("Error: Arguments don't match either 'human' or a computer strategy.");
            System.exit(1);
        }
        
        // If either player uses the ONNX strategy, prompt for a model file
        if (argList.get(0).equals("onnx") || argList.get(1).equals("onnx")) {
            promptForOnnxModel(stage);
        }
        
        FXMLLoader fxmlLoader = new FXMLLoader(App.class.getResource("game-view.fxml"));
        Parent root = fxmlLoader.load();
        GameController controller = fxmlLoader.getController();
        controller.initGame(argList.get(0), argList.get(1));
        
        Scene scene = new Scene(root, 960, 600);
        stage.setTitle("Othello Demo");
        stage.setScene(scene);
        stage.show();
    }
    
    /**
     * Prompts the user to select an ONNX model file or use the default.
     * 
     * @param stage The primary stage for the prompt
     */
    private void promptForOnnxModel(Stage stage) {
        // First, check if the default model exists
        Path defaultModelPath = Paths.get(System.getProperty("user.dir"), "models", "othello.onnx");
        boolean defaultModelExists = Files.exists(defaultModelPath);
        
        // Create dialog text based on whether default model exists
        String dialogText = defaultModelExists ? 
                "Use default ONNX model (othello.onnx) or select a custom one?" : 
                "No default ONNX model found. Please select a model file.";
        
        // Create custom button types with the appropriate text
        ButtonType defaultButton = new ButtonType(defaultModelExists ? "Use Default" : "Select Model");
        ButtonType customButton = new ButtonType(defaultModelExists ? "Select Custom" : "Cancel");
        ButtonType cancelButton = defaultModelExists ? ButtonType.CANCEL : null;
        
        // Create alert with custom buttons
        Alert alert;
        if (defaultModelExists) {
            alert = new Alert(Alert.AlertType.CONFIRMATION, dialogText, 
                    defaultButton, customButton, cancelButton);
        } else {
            alert = new Alert(Alert.AlertType.CONFIRMATION, dialogText, 
                    defaultButton, customButton);
        }
        alert.setTitle("ONNX Model Selection");
        alert.setHeaderText("Select ONNX Model");
        
        Optional<ButtonType> result = alert.showAndWait();
        
        // Handle user choice
        if (!result.isPresent() || (defaultModelExists && result.get() == cancelButton)) {
            // User canceled, exit the application
            System.out.println("ONNX model selection canceled. Exiting.");
            Platform.exit();
            System.exit(0);
        } else if (result.get() == defaultButton) {
            // Use default model or select a model if no default exists
            if (defaultModelExists) {
                System.out.println("Using default ONNX model: " + defaultModelPath);
            } else {
                // If no default model, the "Select Model" button was pressed
                // So proceed to the file selection code below
                selectCustomModel(stage, defaultModelExists, defaultModelPath);
                return;
            }
        } else if (result.get() == customButton) {
            // Select custom model or cancel if no default exists
            if (defaultModelExists) {
                // "Select Custom" was pressed
                selectCustomModel(stage, defaultModelExists, defaultModelPath);
                return;
            } else {
                // "Cancel" was pressed
                System.out.println("No model selected. Exiting.");
                Platform.exit();
                System.exit(0);
            }
        }
    }
    
    /**
     * Helper method to select a custom ONNX model file.
     * 
     * @param stage The primary stage
     * @param defaultModelExists Whether a default model exists
     * @param defaultModelPath The path to the default model
     */
    private void selectCustomModel(Stage stage, boolean defaultModelExists, Path defaultModelPath) {
        // Select custom model
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Select ONNX Model File");
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("ONNX Models", "*.onnx"));
        
        // Set initial directory to project directory
        fileChooser.setInitialDirectory(new File(System.getProperty("user.dir")));
        
        File selectedFile = fileChooser.showOpenDialog(stage);
        
        if (selectedFile != null) {
            try {
                // Create models directory if it doesn't exist
                Path modelsDir = Paths.get(System.getProperty("user.dir"), "models");
                if (!Files.exists(modelsDir)) {
                    Files.createDirectories(modelsDir);
                }
                
                // Copy the selected file to the models directory as othello.onnx
                Path destinationPath = modelsDir.resolve("othello.onnx");
                Files.copy(selectedFile.toPath(), destinationPath, StandardCopyOption.REPLACE_EXISTING);
                
                System.out.println("Using custom ONNX model: " + selectedFile.getPath());
                System.out.println("Copied to: " + destinationPath);
                
                // Set the model path for the strategy factory
                StrategyFactory.setCustomOnnxModelPath(destinationPath.toString());
            } catch (IOException e) {
                System.err.println("Error copying ONNX model: " + e.getMessage());
                e.printStackTrace();
                
                // Show error to user
                Alert errorAlert = new Alert(Alert.AlertType.ERROR);
                errorAlert.setTitle("Error");
                errorAlert.setHeaderText("Failed to copy ONNX model");
                errorAlert.setContentText("Error: " + e.getMessage());
                errorAlert.showAndWait();
                
                Platform.exit();
                System.exit(1);
            }
        } else {
            // No file selected
            if (!defaultModelExists) {
                System.out.println("No model selected. Exiting.");
                Platform.exit();
                System.exit(0);
            } else {
                System.out.println("No custom model selected. Using default.");
            }
        }
    }
    public static void main(String[] args) {
        launch(args);
    }
}