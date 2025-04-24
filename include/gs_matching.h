
#include <cassert>
#include <functional>
#include <iterator>
#include <vector>
#include <algorithm>

template<typename T1, typename T2>
std::vector<int> gale_shapley(std::vector<T1> hospitals, std::vector<T2> residents,
    std::function<double(const T1&, const T2&)> match_distance) {
    std::vector<int> matches(hospitals.size(), -1);
    std::vector<int> reverse_matches(residents.size(), -1);

    std::vector<std::vector<int>> hospital_preferences(hospitals.size());
    std::vector<int> hospital_next(hospitals.size(), 0);
    std::vector<std::vector<int>> resident_preferences(residents.size());

    for (int i = 0; i < hospitals.size(); i++) {
        hospital_preferences[i].resize(residents.size());
        for (int j = 0; j < residents.size(); j++) {
            hospital_preferences[i][j] = j;
        }
        std::sort(
            hospital_preferences[i].begin(), hospital_preferences[i].end(), [&](int r1, int r2) {
                return match_distance(hospitals[i], residents[r1])
                       < match_distance(hospitals[i], residents[r2]);
            });
    }

    for (int i = 0; i < residents.size(); i++) {
        resident_preferences[i].resize(hospitals.size());
        for (int j = 0; j < hospitals.size(); j++) {
            resident_preferences[i][j] = j;
        }
        std::sort(
            resident_preferences[i].begin(), resident_preferences[i].end(), [&](int h1, int h2) {
                return match_distance(hospitals[h1], residents[i])
                       < match_distance(hospitals[h2], residents[i]);
            });
    }

    int unmatched_hospitals = hospitals.size();

    // https://link.springer.com/article/10.1007/BF01934199
    int stop_cond = hospitals.size() > residents.size() ? hospitals.size() - residents.size() : 0;

    while (unmatched_hospitals > stop_cond) {
        int hospital = -1;
        for (int i = 0; i < matches.size(); i++) {
            if (matches[i] == -1) {
                hospital = i;
                break;
            }
        }

        assert(hospital != -1);

        int resident = hospital_preferences[hospital][hospital_next[hospital]];
        hospital_next[hospital]++;

        if (reverse_matches[resident] == -1) {
            matches[hospital] = resident;
            reverse_matches[resident] = hospital;
            unmatched_hospitals--;
        } else {
            auto h_preference = std::find(resident_preferences[resident].begin(),
                resident_preferences[resident].end(), hospital);
            auto current_preference = std::find(resident_preferences[resident].begin(),
                resident_preferences[resident].end(), reverse_matches[resident]);

            if (std::distance(h_preference, current_preference) < 0) {
                matches[reverse_matches[resident]] = -1;
                matches[hospital] = resident;
                reverse_matches[resident] = hospital;
            }
        }
    }
    return matches;
}
