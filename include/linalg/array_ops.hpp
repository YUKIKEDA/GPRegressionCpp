/**
 * @file array_ops.hpp
 * @brief Array1D / Array2D 用の線形代数・要素演算
 */

#pragma once

#include "linalg/array.hpp"
#include <cmath>
#include <cstddef>

namespace gprcpp
{
  namespace linalg
  {

    /** @brief 各行の L2 ノルムの二乗を返す */
    template <typename T>
    Array1D<T> row_norms_squared(const Array2D<T> &X)
    {
      Array1D<T> out(X.rows(), T{0});
      for (std::size_t i = 0; i < X.rows(); ++i)
      {
        T sum = T{0};
        for (std::size_t j = 0; j < X.cols(); ++j)
        {
          T v = X(i, j);
          sum += v * v;
        }
        out(i) = sum;
      }
      return out;
    }

    /** @brief 各列の平均を返す */
    template <typename T>
    Array1D<T> column_means(const Array2D<T> &X)
    {
      if (X.rows() == 0)
      {
        return Array1D<T>();
      }
      Array1D<T> out(X.cols(), T{0});
      for (std::size_t j = 0; j < X.cols(); ++j)
      {
        T sum = T{0};
        for (std::size_t i = 0; i < X.rows(); ++i)
          sum += X(i, j);
        out(j) = sum / static_cast<T>(X.rows());
      }
      return out;
    }

    /** @brief C = A * B^T。A (n,d), B (m,d) -> C (n,m), C(i,j) = sum_f A(i,f)*B(j,f) */
    template <typename T>
    Array2D<T> matmul_abt(const Array2D<T> &A, const Array2D<T> &B)
    {
      const std::size_t n = A.rows();
      const std::size_t m = B.rows();
      const std::size_t d = A.cols();
      if (B.cols() != d)
      {
        throw std::invalid_argument("matmul_abt: A.cols() != B.cols()");
      }
      Array2D<T> C(n, m, T{0});
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
        {
          T sum = T{0};
          for (std::size_t f = 0; f < d; ++f)
            sum += A(i, f) * B(j, f);
          C(i, j) = sum;
        }
      return C;
    }

    /** @brief D(i,j) += v(i) */
    template <typename T>
    void add_to_columns(Array2D<T> &D, const Array1D<T> &v)
    {
      for (std::size_t i = 0; i < D.rows(); ++i)
        for (std::size_t j = 0; j < D.cols(); ++j)
          D(i, j) += v(i);
    }

    /** @brief D(i,j) += v(j) */
    template <typename T>
    void add_to_rows(Array2D<T> &D, const Array1D<T> &v)
    {
      for (std::size_t i = 0; i < D.rows(); ++i)
        for (std::size_t j = 0; j < D.cols(); ++j)
          D(i, j) += v(j);
    }

    /** @brief Frobenius ノルムの二乗: sum_ij (A(i,j) - B(i,j))^2 */
    template <typename T>
    T frobenius_squared_diff(const Array2D<T> &A, const Array2D<T> &B)
    {
      if (A.rows() != B.rows() || A.cols() != B.cols())
      {
        throw std::invalid_argument("frobenius_squared_diff: shape mismatch");
      }
      T sum = T{0};
      for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
        {
          T d = A(i, j) - B(i, j);
          sum += d * d;
        }
      return sum;
    }

    /** @brief 行 i で最小値をとる列インデックス */
    template <typename T>
    std::size_t row_argmin(const Array2D<T> &D, std::size_t i)
    {
      if (D.cols() == 0)
      {
        throw std::invalid_argument("row_argmin: empty row");
      }
      std::size_t j_min = 0;
      T v_min = D(i, 0);
      for (std::size_t j = 1; j < D.cols(); ++j)
      {
        if (D(i, j) < v_min)
        {
          v_min = D(i, j);
          j_min = j;
        }
      }
      return j_min;
    }

    /** @brief 行 i の X と行 j の Y の二乗ユークリッド距離 */
    template <typename T>
    T row_squared_distance(const Array2D<T> &X, std::size_t xi,
                           const Array2D<T> &Y, std::size_t yj)
    {
      if (X.cols() != Y.cols())
      {
        throw std::invalid_argument("row_squared_distance: col mismatch");
      }
      T sum = T{0};
      for (std::size_t f = 0; f < X.cols(); ++f)
      {
        T d = X(xi, f) - Y(yj, f);
        sum += d * d;
      }
      return sum;
    }

    /** @brief 要素ごとに max(x, 0) */
    template <typename T>
    void cwise_max_zero(Array2D<T> &D)
    {
      for (std::size_t i = 0; i < D.rows(); ++i)
        for (std::size_t j = 0; j < D.cols(); ++j)
          if (D(i, j) < T{0})
            D(i, j) = T{0};
    }

    /** @brief 要素ごとに sqrt を適用した新しい行列 */
    inline Array2D<double> sqrt_elementwise(const Array2D<double> &D)
    {
      Array2D<double> out(D.rows(), D.cols());
      for (std::size_t i = 0; i < D.rows(); ++i)
        for (std::size_t j = 0; j < D.cols(); ++j)
          out(i, j) = std::sqrt(D(i, j));
      return out;
    }

  } // namespace linalg
} // namespace gprcpp
