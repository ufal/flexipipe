#pragma once

#include <unicode/unistr.h>
#include <unicode/ustring.h>
#include <unicode/utf8.h>
#include <unicode/utf16.h>
#include <unicode/utypes.h>
#include <unicode/uchar.h>
#include <string>
#include <utility>

namespace flexitag {
namespace unicode {

/**
 * Convert std::string (assumed UTF-8) to ICU UnicodeString
 */
inline icu::UnicodeString to_unicode_string(const std::string& utf8_str) {
    return icu::UnicodeString::fromUTF8(icu::StringPiece(utf8_str.c_str(), utf8_str.length()));
}

/**
 * Convert ICU UnicodeString to std::string (UTF-8)
 */
inline std::string from_unicode_string(const icu::UnicodeString& ustr) {
    std::string result;
    ustr.toUTF8String(result);
    return result;
}

/**
 * Count Unicode characters (code points) in a UTF-8 string
 * Handles UTF-8, UTF-16, and all Unicode characters correctly
 */
inline size_t char_count(const std::string& utf8_str) {
    if (utf8_str.empty()) {
        return 0;
    }
    icu::UnicodeString ustr = to_unicode_string(utf8_str);
    return ustr.countChar32();  // Counts code points (handles surrogates correctly)
}

/**
 * Get the last N Unicode characters from a string
 * Returns UTF-8 encoded result
 */
inline std::string suffix(const std::string& utf8_str, size_t char_count) {
    if (utf8_str.empty() || char_count == 0) {
        return "";
    }
    icu::UnicodeString ustr = to_unicode_string(utf8_str);
    int32_t length = ustr.length();
    int32_t start_pos = std::max(0, static_cast<int32_t>(length - static_cast<int32_t>(char_count)));
    icu::UnicodeString suffix_ustr = ustr.tempSubString(start_pos);
    return from_unicode_string(suffix_ustr);
}

/**
 * Get the first N Unicode characters from a string
 * Returns UTF-8 encoded result
 */
inline std::string prefix(const std::string& utf8_str, size_t char_count) {
    if (utf8_str.empty() || char_count == 0) {
        return "";
    }
    icu::UnicodeString ustr = to_unicode_string(utf8_str);
    int32_t length = ustr.length();
    int32_t prefix_len = std::min(static_cast<int32_t>(char_count), length);
    icu::UnicodeString prefix_ustr = ustr.tempSubString(0, prefix_len);
    return from_unicode_string(prefix_ustr);
}

/**
 * Get substring starting from a Unicode character position (not byte position)
 * Returns UTF-8 encoded result
 */
inline std::string substr_from_char(const std::string& utf8_str, size_t char_start) {
    if (utf8_str.empty() || char_start == 0) {
        return utf8_str;
    }
    icu::UnicodeString ustr = to_unicode_string(utf8_str);
    int32_t length = ustr.length();
    if (static_cast<int32_t>(char_start) >= length) {
        return "";
    }
    icu::UnicodeString substr_ustr = ustr.tempSubString(static_cast<int32_t>(char_start));
    return from_unicode_string(substr_ustr);
}

/**
 * Get Unicode character at position (returns UTF-8 encoded character)
 */
inline std::string char_at(const std::string& utf8_str, size_t char_pos) {
    if (utf8_str.empty()) {
        return "";
    }
    icu::UnicodeString ustr = to_unicode_string(utf8_str);
    int32_t length = ustr.length();
    if (static_cast<int32_t>(char_pos) >= length) {
        return "";
    }
    icu::UnicodeString char_ustr = ustr.tempSubString(static_cast<int32_t>(char_pos), 1);
    return from_unicode_string(char_ustr);
}

/**
 * Get byte range for Unicode character at position
 * Returns (byte_start, byte_length) pair
 */
inline std::pair<size_t, size_t> char_byte_range(const std::string& utf8_str, size_t char_pos) {
    if (utf8_str.empty()) {
        return {0, 0};
    }
    icu::UnicodeString ustr = to_unicode_string(utf8_str);
    int32_t length = ustr.length();
    if (static_cast<int32_t>(char_pos) >= length) {
        return {utf8_str.size(), 0};
    }
    
    // Get the character as UTF-8 to determine byte length
    icu::UnicodeString char_ustr = ustr.tempSubString(static_cast<int32_t>(char_pos), 1);
    std::string char_utf8 = from_unicode_string(char_ustr);
    
    // Find byte position by converting substring up to char_pos
    if (char_pos == 0) {
        return {0, char_utf8.size()};
    }
    icu::UnicodeString prefix_ustr = ustr.tempSubString(0, static_cast<int32_t>(char_pos));
    std::string prefix_utf8 = from_unicode_string(prefix_ustr);
    
    return {prefix_utf8.size(), char_utf8.size()};
}

/**
 * Check if a Unicode character exists in a string (Unicode-aware)
 * Returns true if the character is found as a complete character, not as part of another character
 */
inline bool char_exists_in_string(const std::string& utf8_str, const std::string& char_utf8) {
    if (utf8_str.empty() || char_utf8.empty()) {
        return false;
    }
    icu::UnicodeString ustr = to_unicode_string(utf8_str);
    icu::UnicodeString char_ustr = to_unicode_string(char_utf8);
    
    // Check if character exists (indexOf returns -1 if not found)
    return ustr.indexOf(char_ustr) >= 0;
}

/**
 * Split a pattern string "from_to" into two Unicode characters
 * Handles multi-byte UTF-8 characters correctly
 * Returns pair of (from_char, to_char) as UTF-8 strings
 */
inline std::pair<std::string, std::string> split_pattern(const std::string& pattern_str) {
    if (pattern_str.empty()) {
        return {"", ""};
    }
    
    // Find underscore position (byte-based, but underscore is single-byte)
    size_t underscore_pos = pattern_str.find('_');
    if (underscore_pos == std::string::npos || underscore_pos == 0 || underscore_pos == pattern_str.length() - 1) {
        return {"", ""};
    }
    
    // Convert to UnicodeString to handle multi-byte characters correctly
    icu::UnicodeString pattern_ustr = to_unicode_string(pattern_str);
    
    // Find underscore in Unicode string
    icu::UnicodeString underscore_ustr = icu::UnicodeString::fromUTF8("_");
    int32_t underscore_idx = pattern_ustr.indexOf(underscore_ustr);
    if (underscore_idx < 0 || underscore_idx == 0 || underscore_idx >= pattern_ustr.length() - 1) {
        return {"", ""};
    }
    
    // Extract from_char (everything before underscore)
    icu::UnicodeString from_ustr = pattern_ustr.tempSubString(0, underscore_idx);
    std::string from_char = from_unicode_string(from_ustr);
    
    // Extract to_char (everything after underscore)
    icu::UnicodeString to_ustr = pattern_ustr.tempSubString(underscore_idx + 1);
    std::string to_char = from_unicode_string(to_ustr);
    
    return {from_char, to_char};
}

/**
 * Replace all occurrences of a Unicode character in a string
 * Handles UTF-8, UTF-16, and all Unicode characters correctly
 */
inline std::string replace_char(const std::string& utf8_str, const std::string& from_char_utf8, const std::string& to_char_utf8) {
    if (utf8_str.empty() || from_char_utf8.empty()) {
        return utf8_str;
    }
    icu::UnicodeString ustr = to_unicode_string(utf8_str);
    icu::UnicodeString from_ustr = to_unicode_string(from_char_utf8);
    icu::UnicodeString to_ustr = to_unicode_string(to_char_utf8);
    
    // Replace all occurrences
    ustr.findAndReplace(from_ustr, to_ustr);
    
    return from_unicode_string(ustr);
}

/**
 * Convert string to lowercase (Unicode-aware)
 * Handles all Unicode characters correctly, including multi-byte UTF-8
 */
inline std::string to_lower(const std::string& utf8_str) {
    if (utf8_str.empty()) {
        return utf8_str;
    }
    icu::UnicodeString ustr = to_unicode_string(utf8_str);
    ustr.toLower();  // ICU's Unicode-aware lowercase conversion
    return from_unicode_string(ustr);
}

/**
 * Convert string to uppercase (Unicode-aware)
 * Handles all Unicode characters correctly, including multi-byte UTF-8
 */
inline std::string to_upper(const std::string& utf8_str) {
    if (utf8_str.empty()) {
        return utf8_str;
    }
    icu::UnicodeString ustr = to_unicode_string(utf8_str);
    ustr.toUpper();  // ICU's Unicode-aware uppercase conversion
    return from_unicode_string(ustr);
}

/**
 * Sanitize a string to ensure it's valid UTF-8
 * Replaces invalid sequences with replacement character (U+FFFD)
 */
inline std::string sanitize_utf8(const std::string& str) {
    if (str.empty()) {
        return str;
    }
    // ICU automatically handles invalid UTF-8 by replacing with U+FFFD
    icu::UnicodeString ustr = to_unicode_string(str);
    return from_unicode_string(ustr);
}

/**
 * Convert UTF-16 string to UTF-8
 */
inline std::string utf16_to_utf8(const std::u16string& utf16_str) {
    icu::UnicodeString ustr(reinterpret_cast<const UChar*>(utf16_str.data()), static_cast<int32_t>(utf16_str.length()));
    return from_unicode_string(ustr);
}

/**
 * Convert UTF-8 string to UTF-16
 */
inline std::u16string utf8_to_utf16(const std::string& utf8_str) {
    icu::UnicodeString ustr = to_unicode_string(utf8_str);
    int32_t length = ustr.length();
    std::u16string result;
    result.resize(length);
    // ICU extract method: extract(dest, destCapacity, errorCode)
    UErrorCode status = U_ZERO_ERROR;
    ustr.extract(reinterpret_cast<UChar*>(&result[0]), length, status);
    if (U_FAILURE(status)) {
        // If extraction failed, return empty string
        return std::u16string();
    }
    return result;
}

}  // namespace unicode
}  // namespace flexitag

