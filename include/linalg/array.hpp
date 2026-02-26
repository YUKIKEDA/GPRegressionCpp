/**
 * @file array.hpp
 * @brief 1次元・2次元・3次元配列クラス (Array1D, Array2D, Array3D)
 *
 * Array2D/Array3D は row-major：メモリ上で最後の次元が連続。
 * 行（および高次元では右側のインデックス）方向の走査がキャッシュに優しい。
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace gprcpp
{
  namespace linalg
  {

    /**
     * @brief 1次元配列
     *
     * std::vector に近いインターフェースで、size/operator()/at/data/fill/resize を提供。
     */
    template <typename T>
    class Array1D
    {
    public:
      using value_type = T;
      using size_type = std::size_t;
      using reference = T &;
      using const_reference = const T &;
      using pointer = T *;
      using const_pointer = const T *;

      /** @brief デフォルトコンストラクタ（長さ 0） */
      Array1D() : data_() {}

      /**
       * @brief サイズ指定コンストラクタ
       * @param n 要素数
       * @param value 初期値（省略時は T{}）
       */
      explicit Array1D(size_type n, const T &value = T{}) : data_(n, value) {}

      /** @brief 要素数 */
      size_type size() const { return data_.size(); }

      /** @brief 空かどうか */
      bool empty() const { return data_.empty(); }

      /** @brief 要素アクセス（境界チェックなし） */
      reference operator()(size_type i) { return data_[i]; }
      const_reference operator()(size_type i) const { return data_[i]; }

      /** @brief 要素アクセス（境界チェックあり） */
      reference at(size_type i)
      {
        check_bounds(i);
        return data_[i];
      }
      const_reference at(size_type i) const
      {
        check_bounds(i);
        return data_[i];
      }

      /** @brief 生ポインタ */
      pointer data() { return data_.data(); }
      const_pointer data() const { return data_.data(); }

      /** @brief 全要素を value で埋める */
      void fill(const T &value) { std::fill(data_.begin(), data_.end(), value); }

      /** @brief サイズ変更 */
      void resize(size_type n, const T &value = T{}) { data_.assign(n, value); }

    private:
      std::vector<T> data_;

      void check_bounds(size_type i) const
      {
        if (i >= data_.size())
        {
          throw std::out_of_range(
              "Array1D: index " + std::to_string(i) +
              " out of range [0, " + std::to_string(data_.size()) + ")");
        }
      }
    };

    /**
     * @brief Row-major 2次元配列
     *
     * 要素 (i, j) は内部バッファの index = i * cols() + j に格納される。
     */
    template <typename T>
    class Array2D
    {
    public:
      using value_type = T;
      using size_type = std::size_t;
      using reference = T &;
      using const_reference = const T &;
      using pointer = T *;
      using const_pointer = const T *;

      /** @brief デフォルトコンストラクタ（0x0） */
      Array2D() : rows_(0), cols_(0), data_() {}

      /**
       * @brief サイズ指定コンストラクタ
       * @param rows 行数
       * @param cols 列数
       * @param value 初期値（省略時は T{}）
       */
      Array2D(size_type rows, size_type cols, const T &value = T{})
          : rows_(rows), cols_(cols), data_(rows * cols, value)
      {
      }

      /** @brief 行数 */
      size_type rows() const { return rows_; }

      /** @brief 列数 */
      size_type cols() const { return cols_; }

      /** @brief 要素数 (rows * cols) */
      size_type size() const { return data_.size(); }

      /** @brief 空かどうか */
      bool empty() const { return data_.empty(); }

      /**
       * @brief 要素アクセス（境界チェックなし）
       * @param i 行インデックス [0, rows)
       * @param j 列インデックス [0, cols)
       */
      reference operator()(size_type i, size_type j)
      {
        return data_[index(i, j)];
      }

      const_reference operator()(size_type i, size_type j) const
      {
        return data_[index(i, j)];
      }

      /**
       * @brief 要素アクセス（境界チェックあり）
       */
      reference at(size_type i, size_type j)
      {
        check_bounds(i, j);
        return data_[index(i, j)];
      }

      const_reference at(size_type i, size_type j) const
      {
        check_bounds(i, j);
        return data_[index(i, j)];
      }

      /** @brief 生ポインタ（row-major で連続） */
      pointer data() { return data_.data(); }
      const_pointer data() const { return data_.data(); }

      /** @brief 全要素を value で埋める */
      void fill(const T &value) { std::fill(data_.begin(), data_.end(), value); }

      /**
       * @brief サイズ変更
       * @param new_rows 新しい行数
       * @param new_cols 新しい列数
       * @param value 拡張時に使う初期値
       */
      void resize(size_type new_rows, size_type new_cols, const T &value = T{})
      {
        rows_ = new_rows;
        cols_ = new_cols;
        data_.assign(rows_ * cols_, value);
      }

      /** @brief i 行目の先頭ポインタ（行の連続アクセス用） */
      pointer row_ptr(size_type i) { return data_.data() + i * cols_; }
      const_pointer row_ptr(size_type i) const { return data_.data() + i * cols_; }

    private:
      size_type rows_;
      size_type cols_;
      std::vector<T> data_;

      size_type index(size_type i, size_type j) const { return i * cols_ + j; }

      void check_bounds(size_type i, size_type j) const
      {
        if (i >= rows_ || j >= cols_)
        {
          throw std::out_of_range(
              "Array2D: index (" + std::to_string(i) + ", " + std::to_string(j) +
              ") out of range [" + std::to_string(rows_) + ", " + std::to_string(cols_) + ")");
        }
      }
    };

    /**
     * @brief Row-major 3次元配列
     *
     * 要素 (i, j, k) は index = i * (cols() * depth()) + j * depth() + k に格納。
     * 最後の次元 k が連続で、行・列・深さの順に走査するとキャッシュに優しい。
     */
    template <typename T>
    class Array3D
    {
    public:
      using value_type = T;
      using size_type = std::size_t;
      using reference = T &;
      using const_reference = const T &;
      using pointer = T *;
      using const_pointer = const T *;

      /** @brief デフォルトコンストラクタ（0x0x0） */
      Array3D() : dim0_(0), dim1_(0), dim2_(0), data_() {}

      /**
       * @brief サイズ指定コンストラクタ
       * @param dim0 第1次元（行に相当）
       * @param dim1 第2次元（列に相当）
       * @param dim2 第3次元（深さ）
       * @param value 初期値（省略時は T{}）
       */
      Array3D(size_type dim0, size_type dim1, size_type dim2, const T &value = T{})
          : dim0_(dim0), dim1_(dim1), dim2_(dim2),
            data_(dim0 * dim1 * dim2, value)
      {
      }

      /** @brief 第1次元のサイズ */
      size_type dim0() const { return dim0_; }
      /** @brief 第2次元のサイズ */
      size_type dim1() const { return dim1_; }
      /** @brief 第3次元のサイズ */
      size_type dim2() const { return dim2_; }

      /** @brief 要素数 (dim0 * dim1 * dim2) */
      size_type size() const { return data_.size(); }

      /** @brief 空かどうか */
      bool empty() const { return data_.empty(); }

      /**
       * @brief 要素アクセス（境界チェックなし）
       * @param i 第1次元 [0, dim0)
       * @param j 第2次元 [0, dim1)
       * @param k 第3次元 [0, dim2)
       */
      reference operator()(size_type i, size_type j, size_type k)
      {
        return data_[index(i, j, k)];
      }
      const_reference operator()(size_type i, size_type j, size_type k) const
      {
        return data_[index(i, j, k)];
      }

      /** @brief 要素アクセス（境界チェックあり） */
      reference at(size_type i, size_type j, size_type k)
      {
        check_bounds(i, j, k);
        return data_[index(i, j, k)];
      }
      const_reference at(size_type i, size_type j, size_type k) const
      {
        check_bounds(i, j, k);
        return data_[index(i, j, k)];
      }

      /** @brief 生ポインタ（row-major で連続） */
      pointer data() { return data_.data(); }
      const_pointer data() const { return data_.data(); }

      /** @brief 全要素を value で埋める */
      void fill(const T &value) { std::fill(data_.begin(), data_.end(), value); }

      /**
       * @brief サイズ変更
       */
      void resize(size_type d0, size_type d1, size_type d2, const T &value = T{})
      {
        dim0_ = d0;
        dim1_ = d1;
        dim2_ = d2;
        data_.assign(dim0_ * dim1_ * dim2_, value);
      }

      /** @brief (i, j, *) のスライス先頭ポインタ（第3次元が連続） */
      pointer slice_ptr(size_type i, size_type j)
      {
        return data_.data() + index(i, j, 0);
      }
      const_pointer slice_ptr(size_type i, size_type j) const
      {
        return data_.data() + index(i, j, 0);
      }

    private:
      size_type dim0_;
      size_type dim1_;
      size_type dim2_;
      std::vector<T> data_;

      size_type index(size_type i, size_type j, size_type k) const
      {
        return i * (dim1_ * dim2_) + j * dim2_ + k;
      }

      void check_bounds(size_type i, size_type j, size_type k) const
      {
        if (i >= dim0_ || j >= dim1_ || k >= dim2_)
        {
          throw std::out_of_range(
              "Array3D: index (" + std::to_string(i) + ", " + std::to_string(j) + ", " +
              std::to_string(k) + ") out of range [" + std::to_string(dim0_) + ", " +
              std::to_string(dim1_) + ", " + std::to_string(dim2_) + ")");
        }
      }
    };

  } // namespace linalg
} // namespace gprcpp
