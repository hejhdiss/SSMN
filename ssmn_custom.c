/**
 * SPARSE-STREAM MEMORY NETWORK (SSMN) - C LIBRARY - CUSTOM SPLITTABLE VERSION
 * 
 * Architecture:
 * 1. Sliding Window Attention - O(n·w) local attention (The Eyes)
 * 2. Neural Synaptic Memory - Fast-weight matrix W_f (The Brain)
 * 3. Decaying Latent Blocks - Hybrid attention + synaptic cells
 * 
 * NEW FEATURE: Adjustable static/plastic split from Python!
 * - Set plastic_ratio from 0.0 to 1.0
 * - 0.2 = 20% plastic (default, brain-like)
 * - 0.5 = 50% plastic (balanced)
 * - 0.9 = 90% plastic (highly adaptive)
 * 
 * Compile:
 * Windows: gcc -shared -o ssmn_custom.dll ssmn_custom.c -lm -O3
 * Linux:   gcc -shared -fPIC -o ssmn_custom.so ssmn_custom.c -lm -O3
 * Mac:     gcc -shared -fPIC -o ssmn_custom.dylib ssmn_custom.c -lm -O3
 * 
 * Licensed under GPL V3.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

// ============================================================================
// DATA STRUCTURES
// ============================================================================

typedef struct {
    int input_dim;            // Input dimension
    int hidden_dim;           // Hidden state dimension
    int output_dim;           // Output dimension
    int window_size;          // Sliding window size
    int num_plastic_layers;   // Number of plastic layers
    int num_static_layers;    // Number of static layers
    float plastic_ratio;      // NEW: Ratio of plastic layers (0.0 - 1.0)
    
    // === SLIDING WINDOW ATTENTION ===
    float *query_weights;           // [hidden_dim × hidden_dim]
    float *key_weights;             // [hidden_dim × hidden_dim]
    float *value_weights;           // [hidden_dim × hidden_dim]
    float *attention_output;        // [hidden_dim × hidden_dim]
    float *window_buffer;           // Circular buffer [window_size × hidden_dim]
    int window_pos;                 // Current position in buffer
    int window_fill;                // Number of filled positions
    
    // === NEURAL SYNAPTIC MEMORY (MN Layer) ===
    float *synaptic_weights;        // Fast-weight matrix [hidden_dim × hidden_dim]
    float plasticity_eta;           // η - plasticity rate
    float decay_lambda;             // λ - decay rate
    
    // === STATIC LAYERS ===
    float *static_weights;          // [num_static_layers × hidden_dim × hidden_dim]
    float *static_bias;             // [num_static_layers × hidden_dim]
    
    // === PLASTIC LAYERS (Memory Hubs) ===
    float *plastic_weights;         // [num_plastic_layers × hidden_dim × hidden_dim]
    float *plastic_bias;            // [num_plastic_layers × hidden_dim]
    
    // === INPUT/OUTPUT PROJECTIONS ===
    float *input_projection;        // [input_dim × hidden_dim]
    float *output_projection;       // [hidden_dim × output_dim]
    float *output_bias;             // [output_dim]
    
    // === STATE VARIABLES ===
    float *hidden_state;            // Current hidden state [hidden_dim]
    float *prev_hidden_state;       // Previous hidden state [hidden_dim]
    float *output_state;            // Current output [output_dim]
    
    // === STATISTICS ===
    float avg_synaptic_energy;      // Energy in synaptic weights
    float avg_attention_entropy;    // Entropy of attention distribution
    float synaptic_update_magnitude;// Magnitude of last synaptic update
    
} SSMN;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float tanh_act(float x) {
    return tanhf(x);
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ============================================================================
// INITIALIZATION WITH CUSTOM SPLIT
// ============================================================================

EXPORT SSMN* create_ssmn_custom(
    int input_dim,
    int hidden_dim,
    int output_dim,
    int window_size,
    float plasticity_eta,
    float decay_lambda,
    float plastic_ratio  // NEW PARAMETER: 0.0 to 1.0
) {
    SSMN *net = (SSMN*)malloc(sizeof(SSMN));
    if (!net) return NULL;
    
    net->input_dim = input_dim;
    net->hidden_dim = hidden_dim;
    net->output_dim = output_dim;
    net->window_size = window_size;
    net->plasticity_eta = plasticity_eta;
    net->decay_lambda = decay_lambda;
    net->plastic_ratio = plastic_ratio;
    
    // Validate plastic_ratio
    if (plastic_ratio < 0.0f) plastic_ratio = 0.0f;
    if (plastic_ratio > 1.0f) plastic_ratio = 1.0f;
    
    // CUSTOM SPLIT: User can now control the ratio!
    int total_layers = 6;
    net->num_plastic_layers = (int)(total_layers * plastic_ratio);
    net->num_static_layers = total_layers - net->num_plastic_layers;
    
    // Ensure at least 1 layer of each type (unless ratio is 0.0 or 1.0)
    if (plastic_ratio > 0.0f && net->num_plastic_layers == 0) {
        net->num_plastic_layers = 1;
        net->num_static_layers = total_layers - 1;
    }
    if (plastic_ratio < 1.0f && net->num_static_layers == 0) {
        net->num_static_layers = 1;
        net->num_plastic_layers = total_layers - 1;
    }
    
    // === Allocate Sliding Window Attention ===
    net->query_weights = (float*)calloc(hidden_dim * hidden_dim, sizeof(float));
    net->key_weights = (float*)calloc(hidden_dim * hidden_dim, sizeof(float));
    net->value_weights = (float*)calloc(hidden_dim * hidden_dim, sizeof(float));
    net->attention_output = (float*)calloc(hidden_dim * hidden_dim, sizeof(float));
    net->window_buffer = (float*)calloc(window_size * hidden_dim, sizeof(float));
    net->window_pos = 0;
    net->window_fill = 0;
    
    // === Allocate Synaptic Memory ===
    net->synaptic_weights = (float*)calloc(hidden_dim * hidden_dim, sizeof(float));
    
    // === Allocate Static Layers ===
    if (net->num_static_layers > 0) {
        net->static_weights = (float*)calloc(net->num_static_layers * hidden_dim * hidden_dim, sizeof(float));
        net->static_bias = (float*)calloc(net->num_static_layers * hidden_dim, sizeof(float));
    } else {
        net->static_weights = NULL;
        net->static_bias = NULL;
    }
    
    // === Allocate Plastic Layers ===
    if (net->num_plastic_layers > 0) {
        net->plastic_weights = (float*)calloc(net->num_plastic_layers * hidden_dim * hidden_dim, sizeof(float));
        net->plastic_bias = (float*)calloc(net->num_plastic_layers * hidden_dim, sizeof(float));
    } else {
        net->plastic_weights = NULL;
        net->plastic_bias = NULL;
    }
    
    // === Allocate Input/Output Projections ===
    net->input_projection = (float*)calloc(input_dim * hidden_dim, sizeof(float));
    net->output_projection = (float*)calloc(hidden_dim * output_dim, sizeof(float));
    net->output_bias = (float*)calloc(output_dim, sizeof(float));
    
    // === Allocate States ===
    net->hidden_state = (float*)calloc(hidden_dim, sizeof(float));
    net->prev_hidden_state = (float*)calloc(hidden_dim, sizeof(float));
    net->output_state = (float*)calloc(output_dim, sizeof(float));
    
    // === Initialize Weights (Xavier/He initialization) ===
    float scale_hidden = sqrtf(2.0f / (float)hidden_dim);
    float scale_input = sqrtf(2.0f / (float)input_dim);
    
    // Initialize attention weights
    for (int i = 0; i < hidden_dim * hidden_dim; i++) {
        net->query_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
        net->key_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
        net->value_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
        net->attention_output[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
    }
    
    // Initialize synaptic weights (small values)
    for (int i = 0; i < hidden_dim * hidden_dim; i++) {
        net->synaptic_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }
    
    // Initialize static layers
    if (net->static_weights) {
        for (int i = 0; i < net->num_static_layers * hidden_dim * hidden_dim; i++) {
            net->static_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
        }
    }
    
    // Initialize plastic layers
    if (net->plastic_weights) {
        for (int i = 0; i < net->num_plastic_layers * hidden_dim * hidden_dim; i++) {
            net->plastic_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
        }
    }
    
    // Initialize input projection
    for (int i = 0; i < input_dim * hidden_dim; i++) {
        net->input_projection[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_input;
    }
    
    // Initialize output projection
    for (int i = 0; i < hidden_dim * output_dim; i++) {
        net->output_projection[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
    }
    
    return net;
}

EXPORT void destroy_ssmn_custom(SSMN *net) {
    if (!net) return;
    
    free(net->query_weights);
    free(net->key_weights);
    free(net->value_weights);
    free(net->attention_output);
    free(net->window_buffer);
    free(net->synaptic_weights);
    if (net->static_weights) free(net->static_weights);
    if (net->static_bias) free(net->static_bias);
    if (net->plastic_weights) free(net->plastic_weights);
    if (net->plastic_bias) free(net->plastic_bias);
    free(net->input_projection);
    free(net->output_projection);
    free(net->output_bias);
    free(net->hidden_state);
    free(net->prev_hidden_state);
    free(net->output_state);
    free(net);
}

// ============================================================================
// SLIDING WINDOW ATTENTION
// ============================================================================

static void sliding_window_attention(SSMN *net, float *input, float *output) {
    int H = net->hidden_dim;
    int W = net->window_size;
    
    // Add current input to circular buffer
    memcpy(&net->window_buffer[net->window_pos * H], input, H * sizeof(float));
    net->window_pos = (net->window_pos + 1) % W;
    if (net->window_fill < W) net->window_fill++;
    
    // Compute Query for current input
    float *query = (float*)calloc(H, sizeof(float));
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < H; j++) {
            query[i] += input[j] * net->query_weights[j * H + i];
        }
    }
    
    // Compute attention scores over window
    float *attention_scores = (float*)calloc(net->window_fill, sizeof(float));
    float scale = 1.0f / sqrtf((float)H);
    
    for (int t = 0; t < net->window_fill; t++) {
        float *key_input = &net->window_buffer[t * H];
        
        float score = 0.0f;
        for (int i = 0; i < H; i++) {
            float key_i = 0.0f;
            for (int j = 0; j < H; j++) {
                key_i += key_input[j] * net->key_weights[j * H + i];
            }
            score += query[i] * key_i;
        }
        attention_scores[t] = score * scale;
    }
    
    // Softmax
    float max_score = attention_scores[0];
    for (int t = 1; t < net->window_fill; t++) {
        if (attention_scores[t] > max_score) max_score = attention_scores[t];
    }
    
    float sum_exp = 0.0f;
    for (int t = 0; t < net->window_fill; t++) {
        attention_scores[t] = expf(attention_scores[t] - max_score);
        sum_exp += attention_scores[t];
    }
    
    for (int t = 0; t < net->window_fill; t++) {
        attention_scores[t] /= sum_exp;
    }
    
    // Compute entropy
    float entropy = 0.0f;
    for (int t = 0; t < net->window_fill; t++) {
        if (attention_scores[t] > 1e-8f) {
            entropy -= attention_scores[t] * logf(attention_scores[t]);
        }
    }
    net->avg_attention_entropy = entropy;
    
    // Weighted sum of Values
    memset(output, 0, H * sizeof(float));
    for (int t = 0; t < net->window_fill; t++) {
        float *value_input = &net->window_buffer[t * H];
        float weight = attention_scores[t];
        
        for (int i = 0; i < H; i++) {
            float value_i = 0.0f;
            for (int j = 0; j < H; j++) {
                value_i += value_input[j] * net->value_weights[j * H + i];
            }
            output[i] += weight * value_i;
        }
    }
    
    // Output projection
    float *temp = (float*)calloc(H, sizeof(float));
    memcpy(temp, output, H * sizeof(float));
    memset(output, 0, H * sizeof(float));
    
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < H; j++) {
            output[i] += temp[j] * net->attention_output[j * H + i];
        }
    }
    
    free(query);
    free(attention_scores);
    free(temp);
}

// ============================================================================
// SYNAPTIC MEMORY UPDATE
// ============================================================================

static void update_synaptic_memory(SSMN *net, float *h_current, float *h_prev) {
    int H = net->hidden_dim;
    
    float energy = 0.0f;
    float update_mag = 0.0f;
    
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < H; j++) {
            int idx = i * H + j;
            
            float outer_product = h_current[i] * h_prev[j];
            float delta = net->plasticity_eta * outer_product - net->decay_lambda * net->synaptic_weights[idx];
            
            net->synaptic_weights[idx] += delta;
            update_mag += delta * delta;
            
            // Clip
            if (net->synaptic_weights[idx] > 5.0f) net->synaptic_weights[idx] = 5.0f;
            if (net->synaptic_weights[idx] < -5.0f) net->synaptic_weights[idx] = -5.0f;
            
            energy += net->synaptic_weights[idx] * net->synaptic_weights[idx];
        }
    }
    
    net->avg_synaptic_energy = energy / (float)(H * H);
    net->synaptic_update_magnitude = sqrtf(update_mag / (float)(H * H));
}

// ============================================================================
// FORWARD PASS
// ============================================================================

EXPORT void ssmn_custom_forward(SSMN *net, float *input, float *output) {
    int H = net->hidden_dim;
    
    memcpy(net->prev_hidden_state, net->hidden_state, H * sizeof(float));
    
    // Project input
    float *projected_input = (float*)calloc(H, sizeof(float));
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < net->input_dim; j++) {
            projected_input[i] += input[j] * net->input_projection[j * H + i];
        }
        projected_input[i] = tanh_act(projected_input[i]);
    }
    
    // Sliding window attention
    float *attention_output = (float*)calloc(H, sizeof(float));
    sliding_window_attention(net, projected_input, attention_output);
    
    // Static layers
    float *layer_output = (float*)calloc(H, sizeof(float));
    memcpy(layer_output, attention_output, H * sizeof(float));
    
    for (int layer = 0; layer < net->num_static_layers; layer++) {
        float *temp = (float*)calloc(H, sizeof(float));
        
        for (int i = 0; i < H; i++) {
            float sum = net->static_bias[layer * H + i];
            for (int j = 0; j < H; j++) {
                sum += layer_output[j] * net->static_weights[layer * H * H + j * H + i];
            }
            temp[i] = relu(sum);
        }
        
        memcpy(layer_output, temp, H * sizeof(float));
        free(temp);
    }
    
    // Plastic layers
    for (int layer = 0; layer < net->num_plastic_layers; layer++) {
        float *temp = (float*)calloc(H, sizeof(float));
        
        for (int i = 0; i < H; i++) {
            float sum = net->plastic_bias[layer * H + i];
            for (int j = 0; j < H; j++) {
                sum += layer_output[j] * net->plastic_weights[layer * H * H + j * H + i];
            }
            temp[i] = relu(sum);
        }
        
        update_synaptic_memory(net, temp, layer_output);
        
        // Apply synaptic weights
        float *synaptic_output = (float*)calloc(H, sizeof(float));
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < H; j++) {
                synaptic_output[i] += net->synaptic_weights[i * H + j] * temp[j];
            }
        }
        
        for (int i = 0; i < H; i++) {
            temp[i] = temp[i] + 0.3f * tanh_act(synaptic_output[i]);
        }
        
        memcpy(layer_output, temp, H * sizeof(float));
        free(temp);
        free(synaptic_output);
    }
    
    memcpy(net->hidden_state, layer_output, H * sizeof(float));
    
    // Output projection
    memset(net->output_state, 0, net->output_dim * sizeof(float));
    for (int i = 0; i < net->output_dim; i++) {
        net->output_state[i] = net->output_bias[i];
        for (int j = 0; j < H; j++) {
            net->output_state[i] += layer_output[j] * net->output_projection[j * net->output_dim + i];
        }
    }
    
    if (output) {
        memcpy(output, net->output_state, net->output_dim * sizeof(float));
    }
    
    free(projected_input);
    free(attention_output);
    free(layer_output);
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

EXPORT void ssmn_custom_reset(SSMN *net) {
    memset(net->hidden_state, 0, net->hidden_dim * sizeof(float));
    memset(net->prev_hidden_state, 0, net->hidden_dim * sizeof(float));
    memset(net->window_buffer, 0, net->window_size * net->hidden_dim * sizeof(float));
    net->window_pos = 0;
    net->window_fill = 0;
}

EXPORT void ssmn_custom_reset_synaptic_memory(SSMN *net) {
    memset(net->synaptic_weights, 0, net->hidden_dim * net->hidden_dim * sizeof(float));
}

// ============================================================================
// GETTERS
// ============================================================================

EXPORT void ssmn_custom_get_hidden_state(SSMN *net, float *output) {
    memcpy(output, net->hidden_state, net->hidden_dim * sizeof(float));
}

EXPORT void ssmn_custom_get_synaptic_weights(SSMN *net, float *output) {
    memcpy(output, net->synaptic_weights, net->hidden_dim * net->hidden_dim * sizeof(float));
}

EXPORT float ssmn_custom_get_synaptic_energy(SSMN *net) {
    return net->avg_synaptic_energy;
}

EXPORT float ssmn_custom_get_attention_entropy(SSMN *net) {
    return net->avg_attention_entropy;
}

EXPORT float ssmn_custom_get_synaptic_update_magnitude(SSMN *net) {
    return net->synaptic_update_magnitude;
}

EXPORT int ssmn_custom_get_window_fill(SSMN *net) {
    return net->window_fill;
}

EXPORT int ssmn_custom_get_num_static_layers(SSMN *net) {
    return net->num_static_layers;
}

EXPORT int ssmn_custom_get_num_plastic_layers(SSMN *net) {
    return net->num_plastic_layers;
}

EXPORT float ssmn_custom_get_plastic_ratio(SSMN *net) {
    return net->plastic_ratio;
}

// ============================================================================
// SETTERS
// ============================================================================

EXPORT void ssmn_custom_set_plasticity_eta(SSMN *net, float eta) {
    net->plasticity_eta = eta;
}

EXPORT void ssmn_custom_set_decay_lambda(SSMN *net, float lambda) {
    net->decay_lambda = lambda;
}

// ============================================================================
// INFO
// ============================================================================

EXPORT void ssmn_custom_print_info(SSMN *net) {
    printf("\n=== Sparse-Stream Memory Network (CUSTOM SPLIT) ===\n");
    printf("Input Dim: %d\n", net->input_dim);
    printf("Hidden Dim: %d\n", net->hidden_dim);
    printf("Output Dim: %d\n", net->output_dim);
    printf("Window Size: %d\n", net->window_size);
    printf("Plastic Ratio: %.1f%%\n", net->plastic_ratio * 100.0f);
    printf("Static Layers: %d (%.0f%%)\n", net->num_static_layers, 
           (1.0f - net->plastic_ratio) * 100.0f);
    printf("Plastic Layers: %d (%.0f%%)\n", net->num_plastic_layers, 
           net->plastic_ratio * 100.0f);
    printf("Plasticity η: %.4f\n", net->plasticity_eta);
    printf("Decay λ: %.4f\n", net->decay_lambda);
    printf("\n--- Statistics ---\n");
    printf("Synaptic Energy: %.6f\n", net->avg_synaptic_energy);
    printf("Attention Entropy: %.4f\n", net->avg_attention_entropy);
    printf("Update Magnitude: %.6f\n", net->synaptic_update_magnitude);
    printf("Window Fill: %d/%d\n", net->window_fill, net->window_size);
}