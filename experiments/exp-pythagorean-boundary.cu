#include <stdio.h>
#include <math.h>

// Generate primitive Pythagorean triples up to N
// Test: does a² + b² == c² hold in f32? In f64? Where does it break?
__global__ void test_triple_integrity(int max_side, int *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread tests one triple (m,m+1) where c = m² + (m+1)² 
    // Actually test all triples a² + b² = c² for a in [1, max_side]
    int a = idx + 1;
    if (a > max_side) return;
    
    int fails_f32 = 0, fails_f64 = 0, total = 0;
    
    for (int b = a; b <= max_side; b++) {
        double c2_d = (double)a*a + (double)b*b;
        double c_d = sqrt(c2_d);
        double c_d_rounded = round(c_d);
        
        if (c_d_rounded * c_d_rounded == c2_d) {
            // This IS a Pythagorean triple
            int c = (int)c_d_rounded;
            total++;
            
            // Test f32 preservation
            float af = (float)a, bf = (float)b, cf = (float)c;
            float sum_f32 = af*af + bf*bf;
            float cf_sq = cf*cf;
            if (fabsf(sum_f32 - cf_sq) > 0.001f) fails_f32++;
            
            // Test f64 preservation
            double ad = (double)a, bd = (double)b, cd = (double)c;
            double sum_f64 = ad*ad + bd*bd;
            double cd_sq = cd*cd;
            if (fabs(sum_f64 - cd_sq) > 1e-10) fails_f64++;
        }
    }
    
    results[idx * 3 + 0] = total;
    results[idx * 3 + 1] = fails_f32;
    results[idx * 3 + 2] = fails_f64;
}

int main() {
    int max_side = 10000;
    int N = max_side;
    int *d_res;
    cudaMalloc(&d_res, N * 3 * sizeof(int));
    cudaMemset(d_res, 0, N * 3 * sizeof(int));
    
    test_triple_integrity<<<(N+255)/256, 256>>>(max_side, d_res);
    cudaDeviceSynchronize();
    
    int *h_res = (int*)malloc(N * 3 * sizeof(int));
    cudaMemcpy(h_res, d_res, N * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    
    int total_triples = 0, total_f32_fails = 0, total_f64_fails = 0;
    int first_f32_fail_a = -1, first_f64_fail_a = -1;
    int triples_by_magnitude[6] = {0}; // 0-99, 100-999, 1000-4990, 5000-10000
    int fails_by_magnitude[6] = {0};
    
    for (int i = 0; i < N; i++) {
        int t = h_res[i*3+0], f32 = h_res[i*3+1], f64 = h_res[i*3+2];
        total_triples += t;
        total_f32_fails += f32;
        total_f64_fails += f64;
        
        if (f32 > 0 && first_f32_fail_a < 0) first_f32_fail_a = i + 1;
        if (f64 > 0 && first_f64_fail_a < 0) first_f64_fail_a = i + 1;
        
        int mag = 0;
        int a = i+1;
        if (a >= 100) mag = 1;
        if (a >= 1000) mag = 2;
        if (a >= 5000) mag = 3;
        if (a >= 10000) mag = 4;
        triples_by_magnitude[mag] += t;
        fails_by_magnitude[mag] += f32;
    }
    
    printf("=== Pythagorean Triple Integrity in IEEE 754 ===\n");
    printf("Sides tested: 1 to %d\n\n", max_side);
    printf("Total Pythagorean triples found: %d\n", total_triples);
    printf("f32 failures (a²+b²≠c² in float): %d (%.2f%%)\n", total_f32_fails, 100.0*total_f32_fails/total_triples);
    printf("f64 failures: %d (%.4f%%)\n", total_f64_fails, 100.0*total_f64_fails/total_triples);
    printf("First f32 failure at side a=%d\n", first_f32_fail_a);
    printf("First f64 failure at side a=%d\n\n", first_f64_fail_a);
    
    printf("=== Breakdown by magnitude ===\n");
    const char *ranges[] = {"1-99", "100-999", "1000-4999", "5000-9999"};
    for (int i = 0; i < 4; i++) {
        printf("  %8s: %5d triples, %5d f32 failures (%.1f%%)\n", 
               ranges[i], triples_by_magnitude[i], fails_by_magnitude[i],
               triples_by_magnitude[i] > 0 ? 100.0*fails_by_magnitude[i]/triples_by_magnitude[i] : 0);
    }
    
    printf("\n=== THE BOUNDARY ===\n");
    printf("f32 starts losing Pythagorean triples at side=%d\n", first_f32_fail_a);
    printf("This is where IEEE 754 stops being trustworthy for exact geometry.\n");
    printf("Constraint theory's Pythagorean manifold is correct to reject f32 above this point.\n");
    
    cudaFree(d_res);
    free(h_res);
    return 0;
}
