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
    exports deeplearningjava;
    exports graph.core;
    exports graph.traversal;
    exports graph.search;
    exports util;
}