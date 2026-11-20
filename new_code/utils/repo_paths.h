#ifndef LLM_SIM_UTILS_REPO_PATHS_H
#define LLM_SIM_UTILS_REPO_PATHS_H

// Resolves paths relative to the running executable's own location on disk,
// instead of hardcoding an absolute machine-specific path. Every test binary
// under new_code/build/bin/<target> sits at a fixed depth below the repo
// root (new_code/build/bin/<exe> -> repo root is two levels up from build/),
// so the repo root can be derived at runtime via /proc/self/exe without any
// compile-time injection.

#include <string>
#include <unistd.h>
#include <climits>
#include <cstdlib>

namespace LLM_SIM_PATHS {

// Absolute path to the directory containing the currently-running
// executable (e.g. .../new_code/build/bin).
inline std::string exe_dir() {
    char buf[PATH_MAX];
    ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return ".";
    buf[len] = '\0';
    std::string exe_path(buf);
    auto slash = exe_path.find_last_of('/');
    return (slash == std::string::npos) ? std::string(".") : exe_path.substr(0, slash);
}

// Repo root, derived as exe_dir()/../../.. (bin -> build -> new_code -> repo
// root), canonicalized via realpath() so callers get a clean absolute path
// with no embedded ".." segments -- important for any code that compares
// these paths for exact string equality (e.g. a hash-map keyed by path).
inline const std::string& repo_root() {
    static const std::string root = [] {
        const std::string raw = exe_dir() + "/../../..";
        char resolved[PATH_MAX];
        if (::realpath(raw.c_str(), resolved) != nullptr) return std::string(resolved);
        return raw;
    }();
    return root;
}

// Convenience: repo_root()/new_code/DRAM_sim/DRAM_data
inline const std::string& dram_data_dir() {
    static const std::string dir = repo_root() + "/new_code/DRAM_sim/DRAM_data";
    return dir;
}

// Convenience: repo_root()/inst_gen/<subdir>
inline std::string inst_gen_dir(const std::string& subdir) {
    return repo_root() + "/inst_gen/" + subdir;
}

// Convenience: repo_root()/<relative>, for one-off scratch/fixture paths.
inline std::string repo_path(const std::string& relative) {
    return repo_root() + "/" + relative;
}

} // namespace LLM_SIM_PATHS

#endif // LLM_SIM_UTILS_REPO_PATHS_H
