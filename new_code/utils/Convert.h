#ifndef LLM_SIM_CONVERT_H
#define LLM_SIM_CONVERT_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

class Convert {
    public:
        void execute(std::string input_file, std::string output_file);
    private:
        std::vector<std::string> transform(const std::string& line);
};

#endif