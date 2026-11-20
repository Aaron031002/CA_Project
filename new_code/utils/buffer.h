#ifndef LLM_SIM_BUFFER_H
#define LLM_SIM_BUFFER_H

#include <iostream>
#include <queue>
#include <memory>
#include <systemc.h>


// interface
template <typename T>
class buffer_write_if: virtual public sc_interface {
    public:
        virtual bool nb_write(const T& val) = 0;
        virtual const sc_event& data_read_event() const = 0;
        virtual bool full() const = 0;
};
template <typename T>
class buffer_read_if: virtual public sc_interface {
    public:
        virtual bool nb_read(T& val) = 0;
        virtual const sc_event& data_write_event() const = 0;
        virtual bool empty() const = 0;
};

// Channel
template <typename T>
class buffer: public buffer_read_if<T>, public buffer_write_if<T>, public sc_prim_channel {
    public:
        explicit buffer(sc_module_name name): sc_prim_channel(name), val_(T{}), val_for_write_(T{}) {}
        bool nb_write(const T& val) override {
            // Bug fix (hang investigation,
            // 2026-07-28): must also reject when a write is already staged
            // this same delta-cycle (data_written == true), not just when
            // has_value (the LAST cycle's committed state) is true.
            // has_value only becomes true inside update(), which runs once
            // per delta-cycle after every process has yielded -- so a
            // producer that calls nb_write() more than once in the SAME
            // cycle without an intervening wait() (e.g. a thread whose
            // nb_read()+nb_write() pair on OTHER channels both happen to
            // succeed immediately, looping back for another packet before
            // ever blocking) would see has_value still false on every one
            // of those calls, silently overwriting val_for_write_ each
            // time -- only the LAST same-cycle write survives into the
            // channel; every earlier one is lost with no error, no retry,
            // and no way for the sender to know. Confirmed as the root
            // cause of a real hang: OffChipPacketAdapter::depacketize_thread
            // drained several queued packets in a single delta-cycle
            // (nothing in its own loop forced a wait() between them) and
            // wrote them all to OffChipDRAM::req_port (this channel type),
            // silently dropping all but the last -- the dropped requests'
            // originating cores then waited forever for a response that
            // was never going to be computed.
            if (has_value || data_written) {
                return false;  // buffer is full (committed, or already staged this cycle)
            }
            val_for_write_ = val;
            data_written = true;
            request_update();
            return true;
        }
        bool nb_read(T& val) override {
            if (!has_value) {
                return false;  // buffer is empty
            }
            val = val_;
            data_read = true;
            request_update();
            return true;
        }
        const sc_event& data_read_event() const override {
            return m_data_read_e;
        }
        const sc_event& data_write_event() const override {
            return m_data_write_e;
        }
        bool full() const override {
            return has_value;
        }
        bool empty() const override {
            return !has_value;
        }
        void update() override {
            // Handle write operation
            
            // Handle read operation  
            if (data_written) {
                val_ = val_for_write_;
                has_value = true;
                m_data_write_e.notify(SC_ZERO_TIME);
                data_written = false;
            }
            if (data_read) {
                has_value = false;
                m_data_read_e.notify(SC_ZERO_TIME);
                data_read = false;    
            }
            if(data_read && !data_written)
                val_ = T{};
            
        }

    private:
        T val_, val_for_write_;
        bool data_written = false;
        bool data_read = false;
        bool has_value = false;
        sc_event m_data_read_e;
        sc_event m_data_write_e;
};

// buffer input and output ports
template <typename T>
class buffer_out: public sc_port<buffer_write_if<T>> {
    public:        
        buffer_out(): sc_port<buffer_write_if<T>>() {}
        explicit buffer_out(const char* name): sc_port<buffer_write_if<T>>(name) {}
        bool nb_write(const T& val) { return (*this)->nb_write(val); }
        bool full() const { return (*this)->full(); }
        const sc_event& data_read_event() const { return (*this)->data_read_event(); }
};
template <typename T>
class buffer_in: public sc_port<buffer_read_if<T>> {
    public:
        buffer_in(): sc_port<buffer_read_if<T>>() {}
        explicit buffer_in(const char* name): sc_port<buffer_read_if<T>>(name) {}
        bool nb_read(T& val) { return (*this)->nb_read(val); }
        bool empty() const { return (*this)->empty(); }
        const sc_event& data_write_event() const { return (*this)->data_write_event(); }
};



#endif