#include "householder_qr.h"
#include <malloc.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define RETURN_OK_QR (0)
#define RETURN_NG_QR (1)
#define MEPS_QR (1e-10)
#define INDEX(row, col, dim) ((row) * (dim) + (col))

struct qr_decomp_param_t {
    int32_t dim;
    double *Q; /* 直交行列Q */
    double *R; /* 上三角行列R */
    double *H; /* ハウスホルダー行列H */
    double *u; /* ハウスホルダー変換用のベクトルu */
};

/*
 * \brief 初期化処理
 * \param[in]  size  配列のサイズ
 * \param[in]  val   初期値
 * \param[out] array 初期化対象
 * \retval RETURN_OK_QR 正常終了
 * \retval RETURN_NG_QR 異常終了
 */
static int32_t init_array(int32_t size, double val, double *array);
/*
 * \brief 単位行列の設定
 * \param[in]    dim    次元数
 * \param[inout] matrix 対象の行列
 * \retval RETURN_OK_QR 正常終了
 * \retval RETURN_NG_QR 異常終了
 */
static int32_t create_eye_matrix(int32_t dim, double *matrix);
/*
 * \brief 行列のコピー
 * \param[in]  dim    次元数
 * \param[in]  input  入力行列
 * \param[out] output 出力行列
 * \retval RETURN_OK_QR 正常終了
 * \retval RETURN_NG_QR 異常終了
 */
static int32_t copy_matrix(int32_t dim, double *input, double *output);
/*
 * \brief 行列積の計算
 * \param[in]  dim       次元数
 * \param[in]  left_mat  かけられる行列
 * \param[in]  right_mat かける行列
 * \param[out] out_mat   行列積の結果
 * \retval RETURN_OK_QR 正常終了
 * \retval RETURN_NG_QR 異常終了
 */
static int32_t mul_mm(int32_t dim, double *left_mat, double *right_mat, double *out_mat);
/*
 * \brief 転置処理
 * \param[in]    dim
 * \param[inout] matrix
 * \retval RETURN_OK_QR 正常終了
 * \retval RETURN_NG_QR 異常終了
 */
static int32_t transpose(int32_t dim, double *matrix);
/*
 * \brief ハウスホルダー変換を用いたQR分解
 * \param[inout] param 入力パラメータ
 * \param[in]    work  作業用配列
 * \retval RETURN_OK_QR  正常終了
 * \retval RETURN_OK_QR  異常終了
 */
static int32_t qr_decomposition(struct qr_decomp_param_t *param, double *work);



int32_t QR_method(int32_t dim, double *A, double *lambda, int32_t max_iter) {
    int32_t ret = (int32_t)QR_NG;
    int32_t func_val;
    int32_t iter, idx;
    double err;
    struct qr_decomp_param_t param;
    double *Q = NULL; /* 直交行列 */
    double *R = NULL; /* 上三角行列 */
    double *H = NULL; /* ハウスホルダー変換用の行列 */
    double *u = NULL; /* ハウスホルダー行列計算用のベクトル */
    double *work = NULL; /* 作業用配列 */

    if ((NULL != A) && (NULL != lambda)) {
        Q = (double *)malloc(sizeof(double) * dim * dim);
        R = (double *)malloc(sizeof(double) * dim * dim);
        H = (double *)malloc(sizeof(double) * dim * dim);
        u = (double *)malloc(sizeof(double) * dim);
        work = (double *)malloc(sizeof(double) * dim * dim);
        memset(&param, 0, sizeof(struct qr_decomp_param_t));

        if ((NULL == Q) || (NULL == R) || (NULL == H) || (NULL == u) || (NULL == work)) {
            goto EXIT_QR_METHOD;
        }
        func_val = init_array(dim, 0.0, lambda);
        if ((int32_t)RETURN_OK_QR != func_val) {
            goto EXIT_QR_METHOD;
        }
        param.dim = dim;
        param.Q = Q;
        param.R = R;
        param.H = H;
        param.u = u;

        for (iter = 0; iter < max_iter; iter++) {
            /* QR分解 */
            func_val = copy_matrix(dim, A, R);
            if ((int32_t)RETURN_OK_QR != func_val) {
                goto EXIT_QR_METHOD;
            }
            func_val = qr_decomposition(&param, work);
            if ((int32_t)RETURN_OK_QR != func_val) {
                goto EXIT_QR_METHOD;
            }
            /* 反復計算により行列Aを更新 */
            func_val = mul_mm(dim, param.R, param.Q, A);
            if ((int32_t)RETURN_OK_QR != func_val) {
                goto EXIT_QR_METHOD;
            }
            /* 収束判定 */
            err = 0.0;
            for (idx = 0; idx < dim; idx++) {
                err += fabs(lambda[idx] - A[idx * dim + idx]);
            }
            if ((err / (double)dim) < (double)MEPS_QR) {
                break;
            }
            /* 固有値の設定 */
            for (idx = 0; idx < dim; idx++) {
                lambda[idx] = A[idx * dim + idx];
            }
        }
        ret = (int32_t)QR_OK;
    }
EXIT_QR_METHOD:
    if (NULL != Q) {
        free(Q);
    }
    if (NULL != R) {
        free(R);
    }
    if (NULL != H) {
        free(H);
    }
    if (NULL != u) {
        free(u);
    }
    if (NULL != work) {
        free(work);
    }

    return ret;
}

static int32_t init_array(int32_t size, double val, double *array) {
    int32_t ret = (int32_t)RETURN_NG_QR;
    int32_t i;

    if (NULL != array) {
        for (i = 0; i < size; i++) {
            array[i] = val;
        }
        ret = (int32_t)RETURN_OK_QR;
    }

    return ret;
}

static int32_t create_eye_matrix(int32_t dim, double *matrix) {
    int32_t ret = (int32_t)RETURN_NG_QR;
    int32_t func_val;
    int32_t i;

    if (NULL != matrix) {
        func_val = init_array(dim * dim, 0.0, matrix);

        if ((int32_t)RETURN_OK_QR != func_val) {
            goto EXIT_EYE_MAT;
        }
        for (i = 0; i < dim; i++) {
            matrix[INDEX(i, i, dim)] = 1.0;
        }
        ret = (int32_t)RETURN_OK_QR;
    }
EXIT_EYE_MAT:

    return ret;
}

static int32_t copy_matrix(int32_t dim, double *input, double *output) {
    int32_t ret = (int32_t)RETURN_NG_QR;
    int32_t row, col, idx;

    if ((NULL != input) && (NULL != output)) {
        for (row = 0; row < dim; row++) {
            for (col = 0; col < dim; col++) {
                idx = INDEX(row, col, dim);
                output[idx] = input[idx];
            }
        }
        ret = (int32_t)RETURN_OK_QR;
    }

    return ret;
}

static int32_t mul_mm(int32_t dim, double *left_mat, double *right_mat, double *out_mat) {
    int32_t ret = (int32_t)RETURN_NG_QR;
    int32_t row, col, idx;
    double sum;

    if ((NULL != left_mat) && (NULL != right_mat) && (NULL != out_mat)) {
        for (row = 0; row < dim; row++) {
            for (col = 0; col < dim; col++) {
                sum = 0.0;

                for (idx = 0; idx < dim; idx++) {
                    sum += left_mat[INDEX(row, idx, dim)] * right_mat[INDEX(idx, col, dim)];
                }
                out_mat[INDEX(row, col, dim)] = sum;
            }
        }
        ret = (int32_t)RETURN_OK_QR;
    }

    return ret;
}

static int32_t transpose(int32_t dim, double *matrix) {
    int32_t ret = (int32_t)RETURN_NG_QR;
    int32_t row, col;
    double tmp;

    if (NULL != matrix) {
        for (row = 0; row < dim - 1; row++) {
            for (col = row + 1; col < dim; col++) {
                tmp = matrix[INDEX(row, col, dim)];
                matrix[INDEX(row, col, dim)] = matrix[INDEX(col, row, dim)];
                matrix[INDEX(col, row, dim)] = tmp;
            }
        }
        ret = (int32_t)RETURN_OK_QR;
    }

    return ret;
}

static int32_t qr_decomposition(struct qr_decomp_param_t *param, double *work) {
    int32_t ret = (int32_t)RETURN_NG_QR;
    int32_t func_val;
    int32_t dim;
    int32_t row, col, idx, k;
    double diag, norm;
    double *Q, *R, *H, *u;

    if ((NULL != param) && (NULL != (param->Q)) && (NULL != (param->R)) && (NULL != (param->H)) && (NULL != (param->u)) && (NULL != work)) {
        dim = param->dim;
        Q = param->Q;
        R = param->R;
        H = param->H;
        u = param->u;
        /* 初期化 */
        func_val = create_eye_matrix(dim, Q);
        if ((int32_t)RETURN_OK_QR != func_val) {
            goto EXIT_QR_DEC;
        }

        for (k = 0; k < dim - 1; k++) {
            /* ベクトルxの大きさの計算 */
            diag = 0.0;
            for (col = k; col < dim; col++) {
                idx = INDEX(k, col, dim);
                diag += R[idx] * R[idx];
            }
            diag = sqrt(diag);
            /* uとnormの計算 */
            u[k] = R[INDEX(k, k, dim)] - diag;
            norm = u[k] * u[k];
            for (col = k + 1; col < dim; col++) {
                u[col] = R[INDEX(k, col, dim)];
                norm += u[col] * u[col];
            }
            norm = sqrt(norm);
            if (norm < (double)MEPS_QR) {
                break;
            }
            for (col = k; col < dim; col++) {
                u[col] /= norm;
            }
            /* ハウスホルダー行列の計算 */
            func_val = create_eye_matrix(dim, H);
            if ((int32_t)RETURN_OK_QR != func_val) {
                goto EXIT_QR_DEC;
            }
            for (row = k; row < dim; row++) {
                for (col = k; col < dim; col++) {
                    H[INDEX(row, col, dim)] -= 2.0 * u[row] * u[col];
                }
            }
            /* Rの更新 */
            func_val = mul_mm(dim, H, R, work);
            if ((int32_t)RETURN_OK_QR != func_val) {
                goto EXIT_QR_DEC;
            }
            func_val = copy_matrix(dim, work, R);
            if ((int32_t)RETURN_OK_QR != func_val) {
                goto EXIT_QR_DEC;
            }
            /* Qの更新 */
            func_val = transpose(dim, H);
            if ((int32_t)RETURN_OK_QR != func_val) {
                goto EXIT_QR_DEC;
            }
            func_val = mul_mm(dim, Q, H, work);
            if ((int32_t)RETURN_OK_QR != func_val) {
                goto EXIT_QR_DEC;
            }
            func_val = copy_matrix(dim, work, Q);
            if ((int32_t)RETURN_OK_QR != func_val) {
                goto EXIT_QR_DEC;
            }
        }
        ret = (int32_t)RETURN_OK_QR;
    }
EXIT_QR_DEC:

    return ret;
}
