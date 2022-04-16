#ifndef HOUSEHOLDER_QR_H__
#define HOUSEHOLDER_QR_H__

#include <stdint.h>

#define QR_OK (0)
#define QR_NG (1)

/*
 * \brief QR分解を用いたQR法
 * \param[in]    dim      次元数
 * \parma[inout] A        計算対象の行列
 *                        サイズ：dim * dim
 * \param[inout] lambda   固有値
 *                        サイズ：dim
 * \param[in]    max_iter 最大反復回数
 * \retval QR_OK  正常終了
 * \retval QR_NG  異常終了
 */
int32_t QR_method(int32_t dim, double *A, double *lambda, int32_t max_iter);

#endif