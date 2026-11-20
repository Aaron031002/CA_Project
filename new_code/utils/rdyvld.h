#ifndef LLM_SIM_RDYVLD_H
#define LLM_SIM_RDYVLD_H

#include <iostream>
#include <queue>
#include <systemc.h>

/*
    This code implements blocking and non-blocking ready/valid ports
*/

// ============================================================================
// Read Interface (Consumer side)
// ============================================================================
template <typename T>
class rdyvld_read_if : virtual public sc_interface {
public:
    // Blocking read: waits until data is available
    virtual void ready(T& val) = 0;
    // Non-blocking read: returns true if data was available
    virtual bool nb_ready(T& val) = 0;
    // Check if data is available without consuming
    virtual bool is_valid() const = 0;
    // Event notified when data becomes available
    virtual const sc_event& valid_event() const = 0;
};

// ============================================================================
// Write Interface (Producer side)
// ============================================================================
template <typename T>
class rdyvld_write_if : virtual public sc_interface {
public:
    // Blocking write: waits until consumer takes the data
    virtual void valid(const T& val) = 0;
    // Non-blocking write: returns true if data was accepted
    virtual bool nb_valid(const T& val) = 0;
    // Check if channel is ready to accept data
    virtual bool is_ready() const = 0;
    // Event notified when data is consumed
    virtual const sc_event& ready_event() const = 0;
};


// ============================================================================
// Combined Interface
// ============================================================================
/*
template <typename T>
class rdyvld_inf : public rdyvld_read_if<T>, public rdyvld_write_if<T> {
public:
    virtual void ready(T& val) = 0;
    virtual bool nb_ready(T& val) = 0;
    virtual bool is_valid() const = 0;
    virtual void valid(const T& val) = 0;
    virtual bool nb_valid(const T& val) = 0;
    virtual bool is_ready() const = 0;
    virtual const sc_event& valid_event() const = 0;
    virtual const sc_event& ready_event() const = 0;
};
*/

// ============================================================================
// Ready/Valid Channel Implementation
// ============================================================================
// Protocol:
// Blocking:
//   1. Producer calls valid(data) - sets data and waits for consumer
//   2. Consumer calls ready(data) - waits for valid data, takes it, acknowledges
//   3. Blocking handshake completes when both sides have participated
// Non-blocking:
//   1. Producer calls nb_valid(data) - returns true if channel was free
//   2. Consumer calls nb_ready(data) - returns true if data was available
// ============================================================================
template <typename T>
class rdyvld : public rdyvld_read_if<T>, public rdyvld_write_if<T>, public sc_prim_channel {
public:
    rdyvld(sc_module_name name)
        : sc_prim_channel(name)
        , data_valid_(false)
        , data_taken_(false) {}

    // ========================================================================
    // Blocking Operations
    // ========================================================================
    
    // Producer side: send data and wait for consumer to take it
    void valid(const T& val) override {
        // Set the data and mark as valid
        val_ = val;
        data_valid_ = true;
        request_update();  // Notify that data is available
        
        // Wait for consumer to take the data
        while (!data_taken_) {
            wait(m_data_read_e);
        }
        
        // Reset for next transfer
        data_taken_ = false;
    }

    // Consumer side: wait for valid data and take it
    void ready(T& val) override {
        // Wait for producer to provide valid data
        while (!data_valid_) {
            wait(m_data_written_e);
        }
        
        // Take the data
        val = val_;
        data_valid_ = false;
        data_taken_ = true;
        request_update();  // Notify producer that data was taken
    }

    // ========================================================================
    // Non-blocking Operations
    // ========================================================================
    
    // Non-blocking write: try to send data without waiting
    // Returns true if data was accepted (channel was ready)
    bool nb_valid(const T& val) override {
        // Can only write if no data is pending
        if (data_valid_) {
            return false;
        }
        
        val_ = val;
        data_valid_ = true;
        request_update();
        return true;
    }

    // Non-blocking read: try to receive data without waiting
    // Returns true if data was available
    bool nb_ready(T& val) override {
        if (!data_valid_) {
            return false;
        }
        
        val = val_;
        data_valid_ = false;
        data_taken_ = true;
        request_update();
        return true;
    }

    // Check if data is available (for consumer to poll)
    bool is_valid() const override {
        return data_valid_;
    }

    // Check if channel is ready to accept new data (for producer to poll)
    bool is_ready() const override {
        return !data_valid_;
    }

    // ========================================================================
    // SystemC Primitives
    // ========================================================================
    
    // SystemC update phase
    void update() override {
        if (data_valid_) {
            m_data_written_e.notify(SC_ZERO_TIME);
        }
        if (data_taken_) {
            m_data_read_e.notify(SC_ZERO_TIME);
        }
    }

    const sc_event& valid_event() const override {
        return m_data_written_e;
    }

    const sc_event& ready_event() const override {
        return m_data_read_e;
    }

private:
    T val_;                     // Data storage
    sc_event m_data_written_e;  // Notified when data becomes valid
    sc_event m_data_read_e;     // Notified when data is taken
    bool data_valid_;           // True when data is available
    bool data_taken_;           // True when consumer has taken data
};

// ============================================================================
// FIFO Output Port (for producers to write)
// ============================================================================
template <typename T>
class rdyvld_out : public sc_port<rdyvld_write_if<T>> {
public:
    rdyvld_out() : sc_port<rdyvld_write_if<T>>() {}
    explicit rdyvld_out(const char* name) : sc_port<rdyvld_write_if<T>>(name) {}

    // Blocking write
    void valid(const T& val) { (*this)->valid(val); }
    
    // Non-blocking write
    bool nb_valid(const T& val) { return (*this)->nb_valid(val); }
    
    // Check if ready to send
    bool is_ready() const { return (*this)->is_ready(); }
    
    // Event when data is consumed
    const sc_event& ready_event() const { return (*this)->ready_event(); }
};

// ============================================================================
// FIFO Input Port (for consumers to read)
// ============================================================================
template <typename T>
class rdyvld_in : public sc_port<rdyvld_read_if<T>> {
public:
    rdyvld_in() : sc_port<rdyvld_read_if<T>>() {}
    explicit rdyvld_in(const char* name) : sc_port<rdyvld_read_if<T>>(name) {}

    // Blocking read
    void ready(T& val) { (*this)->ready(val); }
    
    // Non-blocking read
    bool nb_ready(T& val) { return (*this)->nb_ready(val); }
    
    // Check if data is available
    bool is_valid() const { return (*this)->is_valid(); }
    
    // Event when data becomes available
    const sc_event& valid_event() const { return (*this)->valid_event(); }
};

#endif