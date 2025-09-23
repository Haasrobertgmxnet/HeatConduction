#pragma once

#pragma once

#include <chrono>
#include <iostream>

namespace Helper {
    class Timer
    {
    public:
        Timer(const std::string& name = "Default Timer Name") : name(name), outputAtExit(true)
        {
            start = std::chrono::steady_clock::now();
        }

        void setOutputAtExit(bool _outputAtExit) {
            outputAtExit = _outputAtExit;
        }

        std::chrono::steady_clock::time_point getStart() const {
            return start;
        }

        std::chrono::milliseconds getDuration() const {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);
        }

        void printDuration() const {
            std::cout << name << ": Measured time: " << getDuration().count() << " Milliseconds.\n";
        }

        ~Timer()
        {
            if (outputAtExit)
            {
                std::cout << "Destructor of " << name << " called. Measured time: " << getDuration().count() << " Milliseconds.\n";
            }
        }

    private:
        bool outputAtExit{};
        std::string name{};
        std::chrono::steady_clock::time_point start{};
    };
}
