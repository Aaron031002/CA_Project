#include <systemc.h>
#include "fifo.h"

// ============================================================================
// Test Data Structure
// ============================================================================
struct Packet {
    int id;
    int data;
    
    Packet() : id(0), data(0) {}
    Packet(int i, int d) : id(i), data(d) {}
    
    bool operator==(const Packet& other) const {
        return id == other.id && data == other.data;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const Packet& p) {
        os << "Packet{id=" << p.id << ", data=" << p.data << "}";
        return os;
    }
};

// ============================================================================
// Producer Module
// ============================================================================
SC_MODULE(Producer) {
    fifo_out<Packet> out;
    
    int num_packets;
    
    SC_CTOR(Producer) : num_packets(10) {
        SC_THREAD(produce);
    }
    
    void produce() {
        for (int i = 0; i < num_packets; i++) {
            Packet p(i, i * 100);
            std::cout << sc_time_stamp() << " [Producer] Writing: " << p << std::endl;
            out.write(p);
            wait(10, SC_NS);
        }
        std::cout << sc_time_stamp() << " [Producer] Done!" << std::endl;
    }
};

// ============================================================================
// Consumer Module
// ============================================================================
SC_MODULE(Consumer) {
    fifo_in<Packet> in;
    
    int num_packets;
    
    SC_CTOR(Consumer) : num_packets(10) {
        SC_THREAD(consume);
    }
    
    void consume() {
        for (int i = 0; i < num_packets; i++) {
            Packet p;
            in.read(p);
            std::cout << sc_time_stamp() << " [Consumer] Read: " << p << std::endl;
            wait(15, SC_NS);  // Consumer is slower than producer
        }
        std::cout << sc_time_stamp() << " [Consumer] Done!" << std::endl;
    }
};

// ============================================================================
// Test Module for Non-blocking Operations
// ============================================================================
SC_MODULE(NonBlockingTester) {
    fifo_out<int> out;
    fifo_in<int> in;
    
    SC_CTOR(NonBlockingTester) {
        SC_THREAD(test_nonblocking);
    }
    
    void test_nonblocking() {
        std::cout << "\n=== Non-blocking Test ===" << std::endl;
        
        // Test nb_write
        int val = 42;
        bool success = out.nb_write(val);
        std::cout << sc_time_stamp() << " nb_write(42): " << (success ? "SUCCESS" : "FAILED") << std::endl;
        
        wait(SC_ZERO_TIME);  // Allow update phase
        wait(SC_ZERO_TIME);
        
        // Test nb_read
        int read_val;
        success = in.nb_read(read_val);
        std::cout << sc_time_stamp() << " nb_read(): " << (success ? "SUCCESS" : "FAILED");
        if (success) {
            std::cout << ", value=" << read_val;
        }
        std::cout << std::endl;
        
        // Test reading from empty FIFO
        success = in.nb_read(read_val);
        std::cout << sc_time_stamp() << " nb_read() on empty: " << (success ? "SUCCESS" : "FAILED (expected)") << std::endl;
        
        std::cout << "=== Non-blocking Test Done ===\n" << std::endl;
    }
};

// ============================================================================
// Test Module for Limited Size FIFO
// ============================================================================
SC_MODULE(LimitedFifoTester) {
    fifo_out<int> out;
    fifo_in<int> in;
    
    SC_CTOR(LimitedFifoTester) {
        SC_THREAD(test_writer);
        SC_THREAD(test_reader);
    }
    
    void test_writer() {
        std::cout << "\n=== Limited FIFO Test (size=4) ===" << std::endl;
        
        for (int i = 0; i < 8; i++) {
            std::cout << sc_time_stamp() << " [Writer] Writing " << i 
                      << " (free=" << out.num_free() << ", full=" << out.full() << ")" << std::endl;
            out.write(i);
            std::cout << sc_time_stamp() << " [Writer] Wrote " << i << std::endl;
        }
        std::cout << sc_time_stamp() << " [Writer] Done!" << std::endl;
    }
    
    void test_reader() {
        wait(50, SC_NS);  // Start reading after some delay
        
        for (int i = 0; i < 8; i++) {
            int val;
            std::cout << sc_time_stamp() << " [Reader] Reading... (avail=" << in.num_avail() << ")" << std::endl;
            in.read(val);
            std::cout << sc_time_stamp() << " [Reader] Read " << val << std::endl;
            wait(20, SC_NS);
        }
        std::cout << sc_time_stamp() << " [Reader] Done!" << std::endl;
    }
};

// ============================================================================
// Top-level Testbench
// ============================================================================
SC_MODULE(Testbench) {
    // Components for basic test
    Producer producer;
    Consumer consumer;
    fifo<Packet> packet_fifo;
    
    // Components for non-blocking test
    NonBlockingTester nb_tester;
    fifo<int> nb_fifo;
    
    // Components for limited FIFO test
    LimitedFifoTester limited_tester;
    fifo<int> limited_fifo;
    
    SC_CTOR(Testbench) 
        : producer("producer")
        , consumer("consumer")
        , packet_fifo("packet_fifo", false)  // Unlimited FIFO
        , nb_tester("nb_tester")
        , nb_fifo("nb_fifo", false)
        , limited_tester("limited_tester")
        , limited_fifo("limited_fifo", true, 4)  // Limited to 4 elements
    {
        // Connect basic test
        producer.out.bind(packet_fifo);
        consumer.in.bind(packet_fifo);
        
        // Connect non-blocking test
        nb_tester.out.bind(nb_fifo);
        nb_tester.in.bind(nb_fifo);
        
        // Connect limited FIFO test
        limited_tester.out.bind(limited_fifo);
        limited_tester.in.bind(limited_fifo);
    }
};

// ============================================================================
// Main
// ============================================================================
int sc_main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "       FIFO Channel Test Suite         " << std::endl;
    std::cout << "========================================" << std::endl;
    
    Testbench tb("testbench");
    
    sc_start(500, SC_NS);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "         All Tests Completed!          " << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
