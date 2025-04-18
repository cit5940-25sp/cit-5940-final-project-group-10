package util;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Test for the HelloWorld class.
 */
public class HelloWorldTest {

    @Test
    public void testSayHello() {
        assertEquals("Hello, World!", HelloWorld.sayHello());
    }
}