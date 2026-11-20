#include <systemc.h>
#include "rdyvld.h"

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
// Producer Module - sends data via valid signal
// ============================================================================
SC_MODULE(Producer) {
    rdyvld_out<Packet> out;
    
    int num_packets;
    int produced_count;
    
    SC_CTOR(Producer) : num_packets(5), produced_count(0) {
        SC_THREAD(produce);
    }
    
    void produce() {
        for (int i = 0; i < num_packets; i++) {
            Packet p(i, i * 100);
            std::cout << sc_time_stamp() << " [Producer] Sending: " << p << std::endl;
            out.valid(p);
            std::cout << sc_time_stamp() << " [Producer] Sent: " << p << " (consumer acknowledged)" << std::endl;
            produced_count++;
            wait(10, SC_NS);
        }
        std::cout << sc_time_stamp() << " [Producer] Done! Sent " << produced_count << " packets" << std::endl;
    }
};

// ============================================================================
// Consumer Module - receives data via ready signal
// ============================================================================
SC_MODULE(Consumer) {
    rdyvld_in<Packet> in;
    
    int num_packets;
    int consumed_count;
    
    SC_CTOR(Consumer) : num_packets(5), consumed_count(0) {
        SC_THREAD(consume);
    }
    
    void consume() {
        for (int i = 0; i < num_packets; i++) {
            Packet p;
            std::cout << sc_time_stamp() << " [Consumer] Waiting for data..." << std::endl;
            in.ready(p);
            std::cout << sc_time_stamp() << " [Consumer] Received: " << p << std::endl;
            consumed_count++;
            
            // Verify data correctness
            if (p.id != i || p.data != i * 100) {
                std::cout << sc_time_stamp() << " [Consumer] ERROR: Expected id=" << i 
                          << ", data=" << i * 100 << std::endl;
            }
            
            wait(15, SC_NS);  // Consumer is slower than producer
        }
        std::cout << sc_time_stamp() << " [Consumer] Done! Received " << consumed_count << " packets" << std::endl;
    }
};

// ============================================================================
// Fast Consumer - no delay between receives
// ============================================================================
SC_MODULE(FastConsumer) {
    rdyvld_in<int> in;
    
    int num_items;
    int received_count;
    
    SC_CTOR(FastConsumer) : num_items(5), received_count(0) {
        SC_THREAD(consume);
    }
    
    void consume() {
        for (int i = 0; i < num_items; i++) {
            int val;
            in.ready(val);
            std::cout << sc_time_stamp() << " [FastConsumer] Got: " << val << std::endl;
            received_count++;
        }
        std::cout << sc_time_stamp() << " [FastConsumer] Done!" << std::endl;
    }
};

// ============================================================================
// Slow Producer - delays before each send
// ============================================================================
SC_MODULE(SlowProducer) {
    rdyvld_out<int> out;
    
    int num_items;
    int sent_count;
    
    SC_CTOR(SlowProducer) : num_items(5), sent_count(0) {
        SC_THREAD(produce);
    }
    
    void produce() {
        for (int i = 0; i < num_items; i++) {
            wait(20, SC_NS);  // Delay before sending
            std::cout << sc_time_stamp() << " [SlowProducer] Sending: " << i << std::endl;
            out.valid(i);
            std::cout << sc_time_stamp() << " [SlowProducer] Acknowledged: " << i << std::endl;
            sent_count++;
        }
        std::cout << sc_time_stamp() << " [SlowProducer] Done!" << std::endl;
    }
};

// ============================================================================
// Alternating Test - multiple back-to-back transfers
// ============================================================================
SC_MODULE(AlternatingProducer) {
    rdyvld_out<int> out;
    
    SC_CTOR(AlternatingProducer) {
        SC_THREAD(run);
    }
    
    void run() {
        std::cout << "\n=== Alternating Test ===" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << sc_time_stamp() << " [AltProducer] valid(" << i << ")" << std::endl;
            out.valid(i);
            std::cout << sc_time_stamp() << " [AltProducer] ack(" << i << ")" << std::endl;
            // No wait - immediate next transfer
        }
        std::cout << sc_time_stamp() << " [AltProducer] All 10 transfers complete!" << std::endl;
    }
};

SC_MODULE(AlternatingConsumer) {
    rdyvld_in<int> in;
    
    SC_CTOR(AlternatingConsumer) {
        SC_THREAD(run);
    }
    
    void run() {
        for (int i = 0; i < 10; i++) {
            int val;
            in.ready(val);
            std::cout << sc_time_stamp() << " [AltConsumer] got " << val << std::endl;
            if (val != i) {
                std::cout << sc_time_stamp() << " [AltConsumer] ERROR: expected " << i << std::endl;
            }
        }
        std::cout << sc_time_stamp() << " [AltConsumer] Verified all 10 values!" << std::endl;
    }
};

// ============================================================================
// Non-blocking Test Module
// ============================================================================
SC_MODULE(NonBlockingTester) {
    rdyvld_out<int> out;
    rdyvld_in<int> in;
    
    int nb_write_success;
    int nb_read_success;
    int nb_write_fail;
    int nb_read_fail;
    
    SC_CTOR(NonBlockingTester)
        : nb_write_success(0), nb_read_success(0)
        , nb_write_fail(0), nb_read_fail(0)
    {
        SC_THREAD(test_producer);
        SC_THREAD(test_consumer);
    }
    
    void test_producer() {
        std::cout << "\n=== Non-blocking Test ===" << std::endl;
        
        // Test 1: nb_valid when channel is free
        std::cout << sc_time_stamp() << " [NB-Producer] is_ready(): " << out.is_ready() << std::endl;
        bool success = out.nb_valid(42);
        std::cout << sc_time_stamp() << " [NB-Producer] nb_valid(42): " << (success ? "SUCCESS" : "FAILED") << std::endl;
        if (success) nb_write_success++; else nb_write_fail++;
        
        wait(SC_ZERO_TIME);  // Let consumer see the data
        
        // Test 2: nb_valid when data is pending (should fail)
        std::cout << sc_time_stamp() << " [NB-Producer] is_ready(): " << out.is_ready() << std::endl;
        success = out.nb_valid(99);
        std::cout << sc_time_stamp() << " [NB-Producer] nb_valid(99) while busy: " 
                  << (success ? "SUCCESS" : "FAILED (expected)") << std::endl;
        if (!success) nb_write_fail++; else nb_write_success++;
        
        wait(10, SC_NS);  // Wait for consumer to take data
        
        // Test 3: Send multiple values with polling
        for (int i = 0; i < 5; i++) {
            // Poll until ready
            while (!out.is_ready()) {
                wait(1, SC_NS);
            }
            success = out.nb_valid(100 + i);
            std::cout << sc_time_stamp() << " [NB-Producer] nb_valid(" << 100 + i << "): " 
                      << (success ? "SUCCESS" : "FAILED") << std::endl;
            if (success) nb_write_success++; else nb_write_fail++;
        }
        
        std::cout << sc_time_stamp() << " [NB-Producer] Done! Success=" << nb_write_success 
                  << ", Expected Fail=" << nb_write_fail << std::endl;
    }
    
    void test_consumer() {
        wait(SC_ZERO_TIME);  // Let producer go first
        wait(SC_ZERO_TIME);
        
        int val;
        
        // Test 1: nb_ready when data is available
        std::cout << sc_time_stamp() << " [NB-Consumer] is_valid(): " << in.is_valid() << std::endl;
        bool success = in.nb_ready(val);
        std::cout << sc_time_stamp() << " [NB-Consumer] nb_ready(): " 
                  << (success ? "SUCCESS" : "FAILED") << ", val=" << val << std::endl;
        if (success) nb_read_success++; else nb_read_fail++;
        
        // Test 2: nb_ready when no data (should fail)
        success = in.nb_ready(val);
        std::cout << sc_time_stamp() << " [NB-Consumer] nb_ready() on empty: " 
                  << (success ? "SUCCESS" : "FAILED (expected)") << std::endl;
        if (!success) nb_read_fail++; else nb_read_success++;
        
        // Test 3: Receive multiple values with polling
        for (int i = 0; i < 5; i++) {
            // Poll until data available
            while (!in.is_valid()) {
                wait(1, SC_NS);
            }
            success = in.nb_ready(val);
            std::cout << sc_time_stamp() << " [NB-Consumer] nb_ready(): " 
                      << (success ? "SUCCESS" : "FAILED") << ", val=" << val << std::endl;
            if (success) nb_read_success++; else nb_read_fail++;
        }
        
        std::cout << sc_time_stamp() << " [NB-Consumer] Done! Success=" << nb_read_success 
                  << ", Expected Fail=" << nb_read_fail << std::endl;
        std::cout << "=== Non-blocking Test Done ===\n" << std::endl;
    }
};

// ============================================================================
// Polling Producer - uses non-blocking with retry
// ============================================================================
SC_MODULE(PollingProducer) {
    rdyvld_out<int> out;
    
    int sent_count;
    
    SC_CTOR(PollingProducer) : sent_count(0) {
        SC_THREAD(run);
    }
    
    void run() {
        std::cout << "\n=== Polling Mode Test ===" << std::endl;
        for (int i = 0; i < 5; i++) {
            // Keep trying until successful
            int attempts = 0;
            while (!out.nb_valid(200 + i)) {
                attempts++;
                wait(1, SC_NS);
            }
            std::cout << sc_time_stamp() << " [PollingProducer] Sent " << 200 + i 
                      << " after " << attempts << " retries" << std::endl;
            sent_count++;
            wait(5, SC_NS);  // Some delay between sends
        }
        std::cout << sc_time_stamp() << " [PollingProducer] Done!" << std::endl;
    }
};

// ============================================================================
// Polling Consumer - uses non-blocking with retry
// ============================================================================
SC_MODULE(PollingConsumer) {
    rdyvld_in<int> in;
    
    int received_count;
    
    SC_CTOR(PollingConsumer) : received_count(0) {
        SC_THREAD(run);
    }
    
    void run() {
        for (int i = 0; i < 5; i++) {
            int val;
            int attempts = 0;
            while (!in.nb_ready(val)) {
                attempts++;
                wait(1, SC_NS);
            }
            std::cout << sc_time_stamp() << " [PollingConsumer] Got " << val 
                      << " after " << attempts << " polls" << std::endl;
            received_count++;
            wait(10, SC_NS);  // Slower consumer
        }
        std::cout << sc_time_stamp() << " [PollingConsumer] Done!" << std::endl;
        std::cout << "=== Polling Mode Test Done ===\n" << std::endl;
    }
};

// ============================================================================
// Top-level Testbench
// ============================================================================
SC_MODULE(Testbench) {
    // Test 1: Basic producer/consumer with custom struct
    Producer producer;
    Consumer consumer;
    rdyvld<Packet> packet_channel;
    
    // Test 2: Fast consumer waiting for slow producer
    SlowProducer slow_producer;
    FastConsumer fast_consumer;
    rdyvld<int> int_channel;
    
    // Test 3: Back-to-back transfers
    AlternatingProducer alt_producer;
    AlternatingConsumer alt_consumer;
    rdyvld<int> alt_channel;
    
    // Test 4: Non-blocking operations
    NonBlockingTester nb_tester;
    rdyvld<int> nb_channel;
    
    // Test 5: Polling mode (non-blocking with retry)
    PollingProducer polling_producer;
    PollingConsumer polling_consumer;
    rdyvld<int> polling_channel;
    
    SC_CTOR(Testbench)
        : producer("producer")
        , consumer("consumer")
        , packet_channel("packet_channel")
        , slow_producer("slow_producer")
        , fast_consumer("fast_consumer")
        , int_channel("int_channel")
        , alt_producer("alt_producer")
        , alt_consumer("alt_consumer")
        , alt_channel("alt_channel")
        , nb_tester("nb_tester")
        , nb_channel("nb_channel")
        , polling_producer("polling_producer")
        , polling_consumer("polling_consumer")
        , polling_channel("polling_channel")
    {
        // Connect Test 1
        producer.out.bind(packet_channel);
        consumer.in.bind(packet_channel);
        
        // Connect Test 2
        slow_producer.out.bind(int_channel);
        fast_consumer.in.bind(int_channel);
        
        // Connect Test 3
        alt_producer.out.bind(alt_channel);
        alt_consumer.in.bind(alt_channel);
        
        // Connect Test 4
        nb_tester.out.bind(nb_channel);
        nb_tester.in.bind(nb_channel);
        
        // Connect Test 5
        polling_producer.out.bind(polling_channel);
        polling_consumer.in.bind(polling_channel);
    }
};

// ============================================================================
// Main
// ============================================================================
int sc_main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "    Ready/Valid Channel Test Suite     " << std::endl;
    std::cout << "========================================" << std::endl;
    
    Testbench tb("testbench");
    
    sc_start(500, SC_NS);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "         All Tests Completed!          " << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Verify results
    bool all_passed = true;
    
    if (tb.producer.produced_count != 5) {
        std::cout << "FAIL: Producer sent " << tb.producer.produced_count << " instead of 5" << std::endl;
        all_passed = false;
    }
    if (tb.consumer.consumed_count != 5) {
        std::cout << "FAIL: Consumer received " << tb.consumer.consumed_count << " instead of 5" << std::endl;
        all_passed = false;
    }
    if (tb.slow_producer.sent_count != 5) {
        std::cout << "FAIL: SlowProducer sent " << tb.slow_producer.sent_count << " instead of 5" << std::endl;
        all_passed = false;
    }
    if (tb.fast_consumer.received_count != 5) {
        std::cout << "FAIL: FastConsumer received " << tb.fast_consumer.received_count << " instead of 5" << std::endl;
        all_passed = false;
    }
    
    // Verify non-blocking tests
    if (tb.nb_tester.nb_write_success != 6) {
        std::cout << "FAIL: NB write success count " << tb.nb_tester.nb_write_success << " instead of 6" << std::endl;
        all_passed = false;
    }
    if (tb.nb_tester.nb_read_success != 6) {
        std::cout << "FAIL: NB read success count " << tb.nb_tester.nb_read_success << " instead of 6" << std::endl;
        all_passed = false;
    }
    
    // Verify polling tests
    if (tb.polling_producer.sent_count != 5) {
        std::cout << "FAIL: PollingProducer sent " << tb.polling_producer.sent_count << " instead of 5" << std::endl;
        all_passed = false;
    }
    if (tb.polling_consumer.received_count != 5) {
        std::cout << "FAIL: PollingConsumer received " << tb.polling_consumer.received_count << " instead of 5" << std::endl;
        all_passed = false;
    }
    
    if (all_passed) {
        std::cout << "\n*** ALL TESTS PASSED ***" << std::endl;
    } else {
        std::cout << "\n*** SOME TESTS FAILED ***" << std::endl;
        return 1;
    }
    
    return 0;
}
