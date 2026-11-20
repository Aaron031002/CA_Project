#include "Convert.h"

void Convert::execute(std::string input_file, std::string output_file) {
    std::ifstream ifs(input_file);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open input file: " << input_file << std::endl;
        return;
    }

    std::ofstream ofs(output_file);
    if (!ofs.is_open()) {
        std::cerr << "Cannot open output file: " << output_file << std::endl;
        return;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        std::vector<std::string> results = transform(line);
        for (const auto& result : results) {
            ofs << result << std::endl;
        }
    }

    ifs.close();
    ofs.close();
}

std::vector<std::string> Convert::transform(const std::string& line) {
    // TODO: implement your transform logic here
    // Convert one input line into one or more output lines
    return {line};
}