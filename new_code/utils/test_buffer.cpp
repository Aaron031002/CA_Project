#include <iostream>
#include <systemc.h>
#include "buffer.h"

// Test module for buffer functionality
SC_MODULE(buffer_test) {
    // Buffer instance
    buffer<int> buf;
    
    // Producer module
    SC_MODULE(producer) {
        buffer_out<int> out_port;
        
        SC_CTOR(producer) : out_port("out_port") {
            SC_THREAD(produce_data);
        }
        
        void produce_data() {
            for (int i = 0; i < 10; ++i) {
                // Wait if buffer is full
                while (out_port.full()) {
                    wait(out_port.data_read_event());
                }
                
                bool success = out_port.nb_write(i);
                std::cout << sc_time_stamp() << ": Producer wrote " << i 
                         << " (success: " << success << ")" << std::endl;
                
                wait(10, SC_NS);
            }
        }
    };
    
    // Consumer module
    SC_MODULE(consumer) {
        buffer_in<int> in_port;
        
        SC_CTOR(consumer) : in_port("in_port") {
            SC_THREAD(consume_data);
        }
        
        void consume_data() {
            int data;
            for (int i = 0; i < 10; ++i) {
                // Wait if buffer is empty
                while (in_port.empty()) {
                    wait(in_port.data_write_event());
                }
                
                bool success = in_port.nb_read(data);
                std::cout << sc_time_stamp() << ": Consumer read " << data 
                         << " (success: " << success << ")" << std::endl;
                
                wait(15, SC_NS);
            }
        }
    };
    
    producer prod;
    consumer cons;
    
    SC_CTOR(buffer_test) : 
        buf("buffer"),
        prod("producer"),
        cons("consumer") {
        
        // Connect ports to buffer
        prod.out_port(buf);
        cons.in_port(buf);
        
        // Test thread for additional functionality
        SC_THREAD(test_buffer_states);
    }
    
    void test_buffer_states() {
        wait(5, SC_NS);
        
        // Test initial state
        std::cout << "\n=== Testing Buffer States ===" << std::endl;
        std::cout << "Initial buffer empty: " << buf.empty() << std::endl;
        std::cout << "Initial buffer full: " << buf.full() << std::endl;
        
        // Test direct write/read
        wait(50, SC_NS);
        std::cout << "\n=== Testing Direct Buffer Access ===" << std::endl;
        
        int test_val = 999;
        if (!buf.full()) {
            bool write_success = buf.nb_write(test_val);
            std::cout << "Direct write " << test_val << " success: " 
                     << write_success << std::endl;
            
            wait(1, SC_NS); // Wait for update cycle
            
            std::cout << "Buffer full after direct write: " << buf.full() << std::endl;
            std::cout << "Buffer empty after direct write: " << buf.empty() << std::endl;
            
            int read_val;
            bool read_success = buf.nb_read(read_val);
            std::cout << "Direct read " << read_val << " success: " 
                     << read_success << std::endl;
                     
            wait(1, SC_NS); // Wait for update cycle
            
            std::cout << "Buffer full after direct read: " << buf.full() << std::endl;
            std::cout << "Buffer empty after direct read: " << buf.empty() << std::endl;
        }
        
        wait(100, SC_NS);
    }
};

// Test multiple data types
SC_MODULE(type_test) {
    buffer<double> double_buf;
    buffer<std::string> string_buf;
    
    SC_CTOR(type_test) : 
        double_buf("double_buffer"),
        string_buf("string_buffer") {
        SC_THREAD(test_types);
    }
    
    void test_types() {
        std::cout << "\n=== Testing Different Data Types ===" << std::endl;
        
        // Test double buffer
        double d_val = 3.14159;
        bool d_write = double_buf.nb_write(d_val);
        std::cout << "Double write " << d_val << " success: " << d_write << std::endl;
        
        wait(1, SC_NS); // Wait for update cycle
        
        double d_read;
        bool d_read_success = double_buf.nb_read(d_read);
        std::cout << "Double read " << d_read << " success: " << d_read_success << std::endl;
        
        wait(1, SC_NS); // Wait for update cycle
        
        // Test string buffer
        std::string s_val = "Hello SystemC";
        bool s_write = string_buf.nb_write(s_val);
        std::cout << "String write '" << s_val << "' success: " << s_write << std::endl;
        
        wait(1, SC_NS); // Wait for update cycle
        
        std::string s_read;
        bool s_read_success = string_buf.nb_read(s_read);
        std::cout << "String read '" << s_read << "' success: " << s_read_success << std::endl;
    }
};

// Event testing module
SC_MODULE(event_test) {
    buffer<int> event_buf;
    
    SC_CTOR(event_test) : event_buf("event_buffer") {
        SC_THREAD(event_writer);
        SC_THREAD(event_reader);
    }
    
    void event_writer() {
        wait(200, SC_NS);
        std::cout << "\n=== Testing Events ===" << std::endl;
        
        int val = 42;
        event_buf.nb_write(val);
        std::cout << sc_time_stamp() << ": Event test - wrote " << val << std::endl;
    }
    
    void event_reader() {
        wait(190, SC_NS);
        
        // Wait for data_write_event
        std::cout << sc_time_stamp() << ": Event test - waiting for write event" << std::endl;
        wait(event_buf.data_write_event());
        
        int val;
        bool success = event_buf.nb_read(val);
        std::cout << sc_time_stamp() << ": Event test - read " << val 
                 << " after write event (success: " << success << ")" << std::endl;
    }
};

// Error condition testing
SC_MODULE(error_test) {
    buffer<int> err_buf;
    
    SC_CTOR(error_test) : err_buf("error_buffer") {
        SC_THREAD(test_error_conditions);
    }
    
    void test_error_conditions() {
        wait(300, SC_NS);
        std::cout << "\n=== Testing Error Conditions ===" << std::endl;
        
        // Test read from empty buffer
        int val;
        std::cout << "Buffer empty: " << err_buf.empty() << std::endl;
        bool read_empty = err_buf.nb_read(val);
        std::cout << "Read from empty buffer success: " << read_empty 
                 << " (value: " << val << ")" << std::endl;
        
        // Fill buffer
        err_buf.nb_write(123);
        wait(1, SC_NS); // Wait for update cycle
        std::cout << "Buffer full after write: " << err_buf.full() << std::endl;
        
        // Test write to full buffer
        bool write_full = err_buf.nb_write(456);
        std::cout << "Write to full buffer success: " << write_full << std::endl;
        
        // Read and verify
        int read_val;
        bool read_success = err_buf.nb_read(read_val);
        std::cout << "Read value: " << read_val << " (success: " << read_success << ")" << std::endl;
    }
};

int sc_main(int argc, char* argv[]) {
    std::cout << "Starting SystemC Buffer Test" << std::endl;
    
    // Instantiate test modules
    buffer_test test1("buffer_test");
    type_test test2("type_test");
    event_test test3("event_test");
    error_test test4("error_test");
    
    // Run simulation
    sc_start(400, SC_NS);
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "All buffer functionality tests completed" << std::endl;
    std::cout << "Simulation ended at: " << sc_time_stamp() << std::endl;
    
    return 0;
}
