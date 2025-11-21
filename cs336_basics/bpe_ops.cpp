#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <regex>
#include <thread>
#include <mutex>
#include <iostream>
#include <sstream>

namespace py = pybind11;

// 修复点：移除了 '(? :' 中间的空格，改为 '(?:'
// 这是一个标准的非捕获组语法。
const std::regex WORD_PATTERN(R"('(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+)", std::regex_constants::ECMAScript);

// 辅助函数：转义正则特殊字符
std::string regex_escape(const std::string& str) {
    // 转义正则元字符，包括 | ( ) [ ] { } * + ? . ^ $ \ #
    static const std::regex special_chars(R"([-[\]{}()*+?.,\^$|#\s])");
    return std::regex_replace(str, special_chars, R"(\$&)");
}

// 线程工作函数
void worker(const std::string& text, 
            const std::string& st_pattern_str,
            bool has_special_tokens,
            std::unordered_map<std::string, int>& local_counts) {
    
    // Fast Path: 无特殊 Token
    if (!has_special_tokens) {
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), WORD_PATTERN);
        auto words_end = std::sregex_iterator();
        for (auto i = words_begin; i != words_end; ++i) {
            local_counts[i->str()]++;
        }
        return;
    }

    // Slow Path: 有特殊 Token，需要模拟 re.split
    std::regex st_regex(st_pattern_str, std::regex_constants::ECMAScript);
    
    auto st_begin = std::sregex_iterator(text.begin(), text.end(), st_regex);
    auto st_end = std::sregex_iterator();

    size_t last_pos = 0;

    for (auto i = st_begin; i != st_end; ++i) {
        size_t match_pos = i->position();
        size_t match_len = i->length();

        // 处理特殊 Token 左侧的普通文本 gap
        if (match_pos > last_pos) {
            std::string chunk = text.substr(last_pos, match_pos - last_pos);
            // 对 gap 进行 GPT-4 分词
            auto words_begin = std::sregex_iterator(chunk.begin(), chunk.end(), WORD_PATTERN);
            auto words_end = std::sregex_iterator();
            for (auto w = words_begin; w != words_end; ++w) {
                local_counts[w->str()]++;
            }
        }

        // 跳过特殊 Token 本身 (不计入统计)
        last_pos = match_pos + match_len;
    }

    // 处理剩余文本
    if (last_pos < text.length()) {
        std::string chunk = text.substr(last_pos);
        auto words_begin = std::sregex_iterator(chunk.begin(), chunk.end(), WORD_PATTERN);
        auto words_end = std::sregex_iterator();
        for (auto w = words_begin; w != words_end; ++w) {
            local_counts[w->str()]++;
        }
    }
}

// 主函数
py::dict count_tokens_parallel(
    std::string input_path, 
    int num_threads, 
    std::vector<std::string> special_tokens) {

    // 1. 构造特殊 Token 正则: (tok1|tok2|...)
    std::string st_pattern_str;
    bool has_special_tokens = !special_tokens.empty();
    
    if (has_special_tokens) {
        std::ostringstream oss;
        oss << "(";
        for (size_t i = 0; i < special_tokens.size(); ++i) {
            if (i > 0) oss << "|";
            oss << regex_escape(special_tokens[i]);
        }
        oss << ")";
        st_pattern_str = oss.str();
    }

    // 2. 读取文件
    std::ifstream file(input_path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Could not open file");
    
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::string buffer(size, '\0');
    if (!file.read(&buffer[0], size)) throw std::runtime_error("Failed to read file");

    // 3. 切分数据块
    std::vector<std::string> chunks;
    size_t chunk_size = size / num_threads;
    size_t start = 0;
    
    for (int i = 0; i < num_threads; ++i) {
        size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
        
        // 边界调整
        if (i != num_threads - 1) {
            while (end < size && buffer[end] != ' ' && buffer[end] != '\n') {
                end++;
            }
        }
        chunks.push_back(buffer.substr(start, end - start));
        start = end;
    }
    
    // 4. 并行处理
    std::unordered_map<std::string, int> final_counts_cpp;

    {
        py::gil_scoped_release release; 

        std::vector<std::thread> threads;
        std::vector<std::unordered_map<std::string, int>> thread_results(num_threads);

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(
                worker, 
                std::cref(chunks[i]), 
                std::cref(st_pattern_str), 
                has_special_tokens, 
                std::ref(thread_results[i])
            );
        }

        for (auto& t : threads) {
            t.join();
        }

        for (const auto& local_map : thread_results) {
            for (const auto& pair : local_map) {
                final_counts_cpp[pair.first] += pair.second;
            }
        }
    } 

    // 5. 转回 Python
    py::dict result;
    for (const auto& pair : final_counts_cpp) {
        result[py::bytes(pair.first)] = pair.second;
    }

    return result;
}

PYBIND11_MODULE(bpe_ops, m) {
    m.doc() = "Accelerated BPE operations using C++";
    m.def("count_tokens_parallel", &count_tokens_parallel, 
          "Count tokens in parallel",
          py::arg("input_path"), 
          py::arg("num_threads"), 
          py::arg("special_tokens"));
}