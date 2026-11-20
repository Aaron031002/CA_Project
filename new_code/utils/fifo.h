#ifndef LLM_SIM_FIFO_H
#define LLM_SIM_FIFO_H

#include <iostream>
#include <queue>
#include <systemc.h>
#include <climits>

// ============================================================================
// FIFO Write Interface (for producers)
// ============================================================================
template <typename T>
class fifo_write_if : virtual public sc_interface {
public:
    virtual bool nb_write(const T& val) = 0;
    virtual void write(const T& val) = 0;
    virtual int num_free() const = 0;
    virtual bool full() const = 0;
    virtual const sc_event& data_read_event() const = 0;
};

// ============================================================================
// FIFO Read Interface (for consumers)
// ============================================================================
template <typename T>
class fifo_read_if : virtual public sc_interface {
public:
    virtual bool nb_read(T& val) = 0;
    virtual void read(T& val) = 0;
    virtual int num_avail() const = 0;
    virtual bool empty() const = 0;
    virtual const sc_event& data_write_event() const = 0;
};


// ============================================================================
// FIFO Channel Implementation
// ============================================================================
// Thread-safety note:
// - Standard SystemC simulation is single-threaded (cooperative multitasking)
// - All SC_THREADs/SC_METHODs run in the same OS thread via coroutines
// - Therefore, no mutex is needed for normal SystemC usage
// - The update() mechanism with shadow_queue provides delta-cycle safety
// - If external OS threads access this FIFO, additional synchronization is needed
// ============================================================================
template <typename T>
class fifo : public fifo_read_if<T>, public fifo_write_if<T>, public sc_prim_channel {
public:
    explicit fifo(sc_module_name name, const bool limit_flag = false, const int max_size = INT_MAX)
        : sc_prim_channel(name), max_size_(max_size), limit_flag_(limit_flag), m_pending_read(0) {}

    // Non-blocking write: returns true if successful, false if full
    bool nb_write(const T& val) override {
        if (full()) {
            return false;
        }
        shadow_queue_.push(val);
        request_update();
        return true;
    }

    // Non-blocking read: returns true if successful, false if empty
    bool nb_read(T& val) override {
        if (queue_.empty()) {
            return false;
        }
        val = queue_.front();
        queue_.pop();
        m_pending_read++;
        request_update();
        return true;
    }

    // Check if FIFO is empty
    // Note: considers both main queue and pending reads
    bool empty() const override {
        return queue_.empty();
    }

    // Check if FIFO is full
    bool full() const override {
        if (limit_flag_) {
            return static_cast<int>(queue_.size() + shadow_queue_.size()) >= max_size_;
        }
        return false;
    }

    // Blocking read: waits until data is available
    void read(T& val) override {
        while (queue_.empty()) {
            wait(m_data_write_e);
        }
        val = queue_.front();
        queue_.pop();
        m_pending_read++;
        request_update();
    }

    // Blocking write: waits until space is available
    void write(const T& val) override {
        while (full()) {
            wait(m_data_read_e);
        }
        shadow_queue_.push(val);
        request_update();
    }

    // Get number of available elements
    int num_avail() const override {
        return static_cast<int>(queue_.size());
    }

    // Get number of free slots (-1 if unlimited)
    int num_free() const override {
        if (limit_flag_) {
            return max_size_ - static_cast<int>(queue_.size()) - static_cast<int>(shadow_queue_.size());
        }
        return -1;  // Unlimited
    }

    // SystemC update phase: move data from shadow queue to main queue
    // This is called by the SystemC kernel during the update phase,
    // ensuring proper delta-cycle semantics
    void update() override {
        bool data_written = false;
        
        while (!shadow_queue_.empty()) {
            queue_.push(shadow_queue_.front());
            shadow_queue_.pop();
            data_written = true;
        }
        
        if (data_written) {
            m_data_write_e.notify(SC_ZERO_TIME);
        }
        
        if (m_pending_read > 0) {
            m_pending_read = 0;
            m_data_read_e.notify(SC_ZERO_TIME);
        }
    }

    const sc_event& data_read_event() const override {
        return m_data_read_e;
    }

    const sc_event& data_write_event() const override {
        return m_data_write_e;
    }

private:
    const int max_size_;
    const bool limit_flag_;
    std::queue<T> queue_;           // Main queue: visible to readers
    std::queue<T> shadow_queue_;    // Shadow queue: holds writes until update phase
    int m_pending_read;             // Count of pending reads for notification
    sc_event m_data_read_e;         // Notified when data is read (space available)
    sc_event m_data_write_e;        // Notified when data is written (data available)
};

// ============================================================================
// FIFO Output Port (for producers to write)
// ============================================================================
template <typename T>
class fifo_out : public sc_port<fifo_write_if<T>> {
public:
    fifo_out() : sc_port<fifo_write_if<T>>() {}
    explicit fifo_out(const char* name) : sc_port<fifo_write_if<T>>(name) {}

    // Convenience methods that delegate to the interface
    bool nb_write(const T& val) { return (*this)->nb_write(val); }
    void write(const T& val) { (*this)->write(val); }
    int num_free() const { return (*this)->num_free(); }
    bool full() const { return (*this)->full(); }
    const sc_event& data_read_event() const { return (*this)->data_read_event(); }
};

// ============================================================================
// FIFO Input Port (for consumers to read)
// ============================================================================
template <typename T>
class fifo_in : public sc_port<fifo_read_if<T>> {
public:
    fifo_in() : sc_port<fifo_read_if<T>>() {}
    explicit fifo_in(const char* name) : sc_port<fifo_read_if<T>>(name) {}

    // Convenience methods that delegate to the interface
    bool nb_read(T& val) { return (*this)->nb_read(val); }
    void read(T& val) { (*this)->read(val); }
    int num_avail() const { return (*this)->num_avail(); }
    bool empty() const { return (*this)->empty(); }
    const sc_event& data_write_event() const { return (*this)->data_write_event(); }
};

#endif