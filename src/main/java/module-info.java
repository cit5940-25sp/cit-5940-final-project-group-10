module othello {
    requires javafx.controls;
    requires javafx.fxml;

    opens othello to javafx.fxml;
    exports othello;
    exports othello.gui;
    opens othello.gui to javafx.fxml;
    exports othello.gamelogic;
    opens othello.gamelogic to javafx.fxml;
    exports othello.gamelogic.strategies;
    
    // Deep learning framework exports
    exports deeplearningjava;
    exports deeplearningjava.api;
    exports deeplearningjava.core;
    exports deeplearningjava.core.activation;
    exports deeplearningjava.factory;
    exports deeplearningjava.layer;
    exports deeplearningjava.network;
    
    // Graph and utility exports
    exports graph.core;
    exports graph.traversal;
    exports graph.search;
}