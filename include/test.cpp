#include "gs_matching.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Simple struct to represent points in 2D space
struct Point {
    double x, y;
};

// Euclidean distance function
double euclidean_distance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

// Helper function to print matches
void print_matches(const std::vector<int>& matches, const std::vector<Point>& hospitals,
    const std::vector<Point>& residents) {
    std::cout << "Matches (Hospital Index -> Resident Index):" << std::endl;
    for (size_t i = 0; i < matches.size(); ++i) {
        std::cout << "  Hospital " << i << " (" << hospitals[i].x << "," << hospitals[i].y << ")";
        if (matches[i] != -1) {
            std::cout << " -> Resident " << matches[i] << " (" << residents[matches[i]].x << ","
                      << residents[matches[i]].y << ")" << std::endl;
        } else {
            std::cout << " -> Unmatched" << std::endl;
        }
    }
}

// Test Case 1: Simple 1-to-1 matching
void test_case_1() {
    std::cout << "--- Test Case 1: 1 Hospital, 1 Resident ---" << std::endl;
    std::vector<Point> hospitals = {{0, 0}};
    std::vector<Point> residents = {{1, 1}};
    std::vector<int> expected_matches = {0};

    std::vector<int> matches = gale_shapley<Point, Point>(hospitals, residents, euclidean_distance);
    print_matches(matches, hospitals, residents);
    assert(matches == expected_matches);
    std::cout << "Test Case 1 Passed!" << std::endl << std::endl;
}

// Test Case 2: 2 Hospitals, 2 Residents - Clear Preferences
void test_case_2() {
    std::cout << "--- Test Case 2: 2 Hospitals, 2 Residents (Clear Pref) ---" << std::endl;
    std::vector<Point> hospitals = {{0, 0}, {5, 5}};
    std::vector<Point> residents = {{1, 1}, {4, 4}};
    // Expected: H0 -> R0, H1 -> R1
    std::vector<int> expected_matches = {0, 1};

    std::vector<int> matches = gale_shapley<Point, Point>(hospitals, residents, euclidean_distance);
    print_matches(matches, hospitals, residents);
    assert(matches == expected_matches);
    std::cout << "Test Case 2 Passed!" << std::endl << std::endl;
}

// Test Case 3: 3 Hospitals, 3 Residents - More Complex
void test_case_3() {
    std::cout << "--- Test Case 3: 3 Hospitals, 3 Residents (Complex) ---" << std::endl;
    std::vector<Point> hospitals = {{0, 0}, {1, 3}, {4, 1}};  // H0, H1, H2
    std::vector<Point> residents = {{1, 1}, {3, 2}, {0, 2}};  // R0, R1, R2
    // Distances:
    // H0: R0(1.41), R1(3.61), R2(2.00) -> Pref: R0, R2, R1
    // H1: R0(2.00), R1(2.83), R2(1.41) -> Pref: R2, R0, R1
    // H2: R0(3.00), R1(1.41), R2(4.12) -> Pref: R1, R0, R2
    // R0: H0(1.41), H1(2.00), H2(3.00) -> Pref: H0, H1, H2
    // R1: H0(3.61), H1(2.83), H2(1.41) -> Pref: H2, H1, H0
    // R2: H0(2.00), H1(1.41), H2(4.12) -> Pref: H1, H0, H2
    // Expected matching (stable): H0->R0, H1->R2, H2->R1
    std::vector<int> expected_matches = {0, 2, 1};

    std::vector<int> matches = gale_shapley<Point, Point>(hospitals, residents, euclidean_distance);
    print_matches(matches, hospitals, residents);
    assert(matches == expected_matches);
    std::cout << "Test Case 3 Passed!" << std::endl << std::endl;
}

// Test Case 4: More Hospitals than Residents
void test_case_4() {
    std::cout << "--- Test Case 4: 3 Hospitals, 2 Residents ---" << std::endl;
    std::vector<Point> hospitals = {{0, 0}, {5, 0}, {0, 5}};  // H0, H1, H2
    std::vector<Point> residents = {{1, 1}, {4, 1}};          // R0, R1
    // Distances:
    // H0: R0(1.41), R1(4.12) -> Pref: R0, R1
    // H1: R0(4.12), R1(1.41) -> Pref: R1, R0
    // H2: R0(5.10), R1(5.83) -> Pref: R0, R1
    // R0: H0(1.41), H1(4.12), H2(5.10) -> Pref: H0, H1, H2
    // R1: H1(1.41), H0(4.12), H2(5.83) -> Pref: H1, H0, H2
    // Expected: H0->R0, H1->R1, H2 unmatched
    std::vector<int> expected_matches = {0, 1, -1};

    std::vector<int> matches = gale_shapley<Point, Point>(hospitals, residents, euclidean_distance);
    print_matches(matches, hospitals, residents);
    assert(matches == expected_matches);
    std::cout << "Test Case 4 Passed!" << std::endl << std::endl;
}

// Test Case 5: More Residents than Hospitals (Current code won't match)
void test_case_5() {
    std::cout << "--- Test Case 5: 2 Hospitals, 4 Residents ---" << std::endl;
    std::vector<Point> hospitals = {{0, 0}, {10, 10}};                // H0, H1
    std::vector<Point> residents = {{1, 1}, {2, 2}, {8, 8}, {9, 9}};  // R0, R1, R2, R3
    // Expected: H0 unmatched, H1 unmatched (due to loop condition)
    std::vector<int> expected_matches = {0, 3};

    std::vector<int> matches = gale_shapley<Point, Point>(hospitals, residents, euclidean_distance);
    print_matches(matches, hospitals, residents);
    assert(matches == expected_matches);
    std::cout << "Test Case 5 Passed! (As expected with current logic)" << std::endl << std::endl;
}

int main() {
    test_case_1();
    test_case_2();
    test_case_3();
    test_case_4();
    test_case_5();

    std::cout << "All Gale-Shapely tests passed!" << std::endl;
    return 0;
}
