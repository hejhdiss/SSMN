/**
 * TEXT-NATIVE SPARSE-STREAM MEMORY NETWORK (Text-Native SSMN) - C LIBRARY
 * 
 * Architecture combines:
 * 1. Neural Semantic Encoder - Converts tokens to "thought embeddings"
 * 2. Sliding Window Attention - Local O(n·w) attention for immediate context
 * 3. Synaptic Memory Layer - Fast-weight matrix W_f for long-term dependencies
 * 4. Internal Recurrent Chat - Model "re-reads" its synaptic state
 * 
 * Key Innovation: Language and Memory are unified - the model stores geometric
 * relationships between concepts in active synapses, not just word vectors.
 * 
 * Synaptic Update: ΔW_f = Gate(Importance) · [η(h_t ⊗ h_{t-1}) - λW_f]
 * 
 * Compile:
 * Windows: gcc -shared -o text_native_ssmn.dll text_native_ssmn.c -lm -O3
 * Linux:   gcc -shared -fPIC -o text_native_ssmn.so text_native_ssmn.c -lm -O3
 * Mac:     gcc -shared -fPIC -o text_native_ssmn.dylib text_native_ssmn.c -lm -O3
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
    int vocab_size;           // Vocabulary size
    int embed_dim;            // Embedding dimension
    int hidden_dim;           // Hidden state dimension
    int window_size;          // Sliding window size
    int num_plastic_layers;   // Number of plastic synaptic layers (20% of total)
    int num_static_layers;    // Number of static layers (80% of total)
    
    // === NEURAL SEMANTIC ENCODER ===
    // Converts tokens to "thought embeddings" capturing intent
    float *semantic_embedding;      // [vocab_size × embed_dim]
    float *intent_projection;       // [embed_dim × hidden_dim]
    float *semantic_bias;           // [hidden_dim]
    
    // === SLIDING WINDOW ATTENTION (Locality First) ===
    float *query_weights;           // [hidden_dim × hidden_dim]
    float *key_weights;             // [hidden_dim × hidden_dim]
    float *value_weights;           // [hidden_dim × hidden_dim]
    float *attention_output;        // [hidden_dim × hidden_dim]
    float *window_buffer;           // Circular buffer [window_size × hidden_dim]
    int window_pos;                 // Current position in circular buffer
    int window_fill;                // How many positions are filled
    
    // === SYNAPTIC MEMORY LAYER (Fast Weights) ===
    // W_f updates during forward pass - stores compressed semantic anchors
    float *synaptic_weights;        // [hidden_dim × hidden_dim] per plastic layer
    float *importance_gate;         // [hidden_dim] - determines what to store
    float *semantic_anchors;        // [hidden_dim] - key semantic points
    float plasticity_eta;           // η - learning rate for synaptic update
    float decay_lambda;             // λ - decay rate for forgetting
    
    // === STATIC LAYERS (80% - Grammar & Logic) ===
    float *static_weights;          // [num_static_layers × hidden_dim × hidden_dim]
    float *static_bias;             // [num_static_layers × hidden_dim]
    
    // === PLASTIC LAYERS (20% - Memory Hubs) ===
    float *plastic_weights;         // [num_plastic_layers × hidden_dim × hidden_dim]
    float *plastic_bias;            // [num_plastic_layers × hidden_dim]
    
    // === STATE VARIABLES ===
    float *hidden_state;            // Current hidden state [hidden_dim]
    float *prev_hidden_state;       // Previous hidden state [hidden_dim]
    float *internal_chat_state;     // State for internal recurrent chat [hidden_dim]
    float *output_logits;           // Output logits [vocab_size]
    
    // === STATISTICS ===
    float avg_importance;           // Average importance gate activation
    float avg_synaptic_energy;      // Energy in synaptic weights
    float avg_attention_entropy;    // Entropy of attention distribution
    
} TextNativeSSMN;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float tanh_act(float x) {
    return tanhf(x);
}

static inline float softmax_stable(float *x, int size, int idx) {
    // Compute max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += expf(x[i] - max_val);
    }
    
    return expf(x[idx] - max_val) / sum;
}

static void normalize_vector(float *vec, int size) {
    float norm = 0.0f;
    for (int i = 0; i < size; i++) {
        norm += vec[i] * vec[i];
    }
    norm = sqrtf(norm + 1e-8f);
    
    for (int i = 0; i < size; i++) {
        vec[i] /= norm;
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

EXPORT TextNativeSSMN* create_text_native_ssmn(
    int vocab_size,
    int embed_dim,
    int hidden_dim,
    int window_size,
    float plasticity_eta,
    float decay_lambda
) {
    TextNativeSSMN *net = (TextNativeSSMN*)malloc(sizeof(TextNativeSSMN));
    if (!net) return NULL;
    
    net->vocab_size = vocab_size;
    net->embed_dim = embed_dim;
    net->hidden_dim = hidden_dim;
    net->window_size = window_size;
    net->plasticity_eta = plasticity_eta;
    net->decay_lambda = decay_lambda;
    
    // 80% static, 20% plastic (brain-inspired)
    int total_layers = 6;
    net->num_static_layers = (int)(total_layers * 0.8f);
    net->num_plastic_layers = total_layers - net->num_static_layers;
    
    // === Allocate Semantic Encoder ===
    net->semantic_embedding = (float*)calloc(vocab_size * embed_dim, sizeof(float));
    net->intent_projection = (float*)calloc(embed_dim * hidden_dim, sizeof(float));
    net->semantic_bias = (float*)calloc(hidden_dim, sizeof(float));
    
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
    net->importance_gate = (float*)calloc(hidden_dim, sizeof(float));
    net->semantic_anchors = (float*)calloc(hidden_dim, sizeof(float));
    
    // === Allocate Static Layers ===
    net->static_weights = (float*)calloc(net->num_static_layers * hidden_dim * hidden_dim, sizeof(float));
    net->static_bias = (float*)calloc(net->num_static_layers * hidden_dim, sizeof(float));
    
    // === Allocate Plastic Layers ===
    net->plastic_weights = (float*)calloc(net->num_plastic_layers * hidden_dim * hidden_dim, sizeof(float));
    net->plastic_bias = (float*)calloc(net->num_plastic_layers * hidden_dim, sizeof(float));
    
    // === Allocate States ===
    net->hidden_state = (float*)calloc(hidden_dim, sizeof(float));
    net->prev_hidden_state = (float*)calloc(hidden_dim, sizeof(float));
    net->internal_chat_state = (float*)calloc(hidden_dim, sizeof(float));
    net->output_logits = (float*)calloc(vocab_size, sizeof(float));
    
    // === Initialize Weights (Xavier/He) ===
    float scale_embed = sqrtf(2.0f / (float)embed_dim);
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        net->semantic_embedding[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_embed;
    }
    
    float scale_hidden = sqrtf(2.0f / (float)hidden_dim);
    
    // Initialize attention weights
    for (int i = 0; i < hidden_dim * hidden_dim; i++) {
        net->query_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
        net->key_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
        net->value_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
        net->attention_output[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
    }
    
    // Initialize intent projection
    for (int i = 0; i < embed_dim * hidden_dim; i++) {
        net->intent_projection[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
    }
    
    // Initialize synaptic weights (small values)
    for (int i = 0; i < hidden_dim * hidden_dim; i++) {
        net->synaptic_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }
    
    // Initialize static layers
    for (int i = 0; i < net->num_static_layers * hidden_dim * hidden_dim; i++) {
        net->static_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
    }
    
    // Initialize plastic layers
    for (int i = 0; i < net->num_plastic_layers * hidden_dim * hidden_dim; i++) {
        net->plastic_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_hidden;
    }
    
    return net;
}

EXPORT void destroy_text_native_ssmn(TextNativeSSMN *net) {
    if (!net) return;
    
    free(net->semantic_embedding);
    free(net->intent_projection);
    free(net->semantic_bias);
    free(net->query_weights);
    free(net->key_weights);
    free(net->value_weights);
    free(net->attention_output);
    free(net->window_buffer);
    free(net->synaptic_weights);
    free(net->importance_gate);
    free(net->semantic_anchors);
    free(net->static_weights);
    free(net->static_bias);
    free(net->plastic_weights);
    free(net->plastic_bias);
    free(net->hidden_state);
    free(net->prev_hidden_state);
    free(net->internal_chat_state);
    free(net->output_logits);
    free(net);
}

// ============================================================================
// NEURAL SEMANTIC ENCODER
// ============================================================================

static void semantic_encode(TextNativeSSMN *net, int token_id, float *output) {
    // Step 1: Lookup embedding
    float *embedding = &net->semantic_embedding[token_id * net->embed_dim];
    
    // Step 2: Project to "thought embedding" capturing intent
    for (int i = 0; i < net->hidden_dim; i++) {
        float sum = net->semantic_bias[i];
        for (int j = 0; j < net->embed_dim; j++) {
            sum += embedding[j] * net->intent_projection[j * net->hidden_dim + i];
        }
        output[i] = tanh_act(sum);
    }
}

// ============================================================================
// SLIDING WINDOW ATTENTION
// ============================================================================

static void sliding_window_attention(TextNativeSSMN *net, float *input, float *output) {
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
    
    // Compute attention over window
    float *attention_scores = (float*)calloc(net->window_fill, sizeof(float));
    float scale = 1.0f / sqrtf((float)H);
    
    for (int t = 0; t < net->window_fill; t++) {
        float *key_input = &net->window_buffer[t * H];
        
        // Compute Key
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
    
    // Softmax over attention scores
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
    
    // Compute attention entropy (for statistics)
    float entropy = 0.0f;
    for (int t = 0; t < net->window_fill; t++) {
        if (attention_scores[t] > 1e-8f) {
            entropy -= attention_scores[t] * logf(attention_scores[t]);
        }
    }
    net->avg_attention_entropy = entropy;
    
    // Compute weighted sum of Values
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
    
    // Apply output projection
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
// SYNAPTIC MEMORY UPDATE (The Core Innovation)
// ============================================================================

static void update_synaptic_memory(TextNativeSSMN *net, float *h_current, float *h_prev) {
    int H = net->hidden_dim;
    
    // Compute importance gate: which neurons carry semantic anchors?
    // Gate = σ(|h_current - h_prev|) - identifies significant changes
    float avg_importance = 0.0f;
    for (int i = 0; i < H; i++) {
        float diff = fabsf(h_current[i] - h_prev[i]);
        net->importance_gate[i] = sigmoid(diff * 10.0f - 2.0f); // Threshold around 0.2
        avg_importance += net->importance_gate[i];
    }
    net->avg_importance = avg_importance / (float)H;
    
    // Identify semantic anchors (high-magnitude activations)
    for (int i = 0; i < H; i++) {
        net->semantic_anchors[i] = fabsf(h_current[i]) > 0.5f ? 1.0f : 0.0f;
    }
    
    // Synaptic Update: ΔW_f = Gate(Importance) · [η(h_t ⊗ h_prev) - λW_f]
    float energy = 0.0f;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < H; j++) {
            int idx = i * H + j;
            
            // Outer product: h_current ⊗ h_prev
            float outer_product = h_current[i] * h_prev[j];
            
            // Gated plasticity with semantic anchors
            float gate = net->importance_gate[i] * net->semantic_anchors[j];
            
            // Hebbian update with decay
            float delta = gate * (net->plasticity_eta * outer_product - net->decay_lambda * net->synaptic_weights[idx]);
            
            net->synaptic_weights[idx] += delta;
            
            // Clip to prevent explosion
            if (net->synaptic_weights[idx] > 5.0f) net->synaptic_weights[idx] = 5.0f;
            if (net->synaptic_weights[idx] < -5.0f) net->synaptic_weights[idx] = -5.0f;
            
            energy += net->synaptic_weights[idx] * net->synaptic_weights[idx];
        }
    }
    net->avg_synaptic_energy = energy / (float)(H * H);
}

// ============================================================================
// INTERNAL RECURRENT CHAT
// ============================================================================

static void internal_recurrent_chat(TextNativeSSMN *net, float *hidden, float *output) {
    int H = net->hidden_dim;
    
    // The model "re-reads" its synaptic state
    // output = tanh(W_f · hidden)
    memset(output, 0, H * sizeof(float));
    
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < H; j++) {
            output[i] += net->synaptic_weights[i * H + j] * hidden[j];
        }
        output[i] = tanh_act(output[i]);
    }
    
    // Store as internal chat state
    memcpy(net->internal_chat_state, output, H * sizeof(float));
}

// ============================================================================
// FORWARD PASS
// ============================================================================

EXPORT void text_native_ssmn_forward(TextNativeSSMN *net, int token_id, float *output_probs) {
    int H = net->hidden_dim;
    
    // Save previous hidden state
    memcpy(net->prev_hidden_state, net->hidden_state, H * sizeof(float));
    
    // 1. Neural Semantic Encoder: token -> thought embedding
    float *thought_embedding = (float*)calloc(H, sizeof(float));
    semantic_encode(net, token_id, thought_embedding);
    
    // 2. Sliding Window Attention: local context
    float *attention_output = (float*)calloc(H, sizeof(float));
    sliding_window_attention(net, thought_embedding, attention_output);
    
    // 3. Process through Static Layers (80%)
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
    
    // 4. Process through Plastic Layers (20%) with Synaptic Memory
    for (int layer = 0; layer < net->num_plastic_layers; layer++) {
        float *temp = (float*)calloc(H, sizeof(float));
        
        // Standard feedforward
        for (int i = 0; i < H; i++) {
            float sum = net->plastic_bias[layer * H + i];
            for (int j = 0; j < H; j++) {
                sum += layer_output[j] * net->plastic_weights[layer * H * H + j * H + i];
            }
            temp[i] = relu(sum);
        }
        
        // Update synaptic memory
        update_synaptic_memory(net, temp, layer_output);
        
        // Internal recurrent chat - re-read synaptic state
        float *chat_output = (float*)calloc(H, sizeof(float));
        internal_recurrent_chat(net, temp, chat_output);
        
        // Combine: temp = temp + chat_output (residual)
        for (int i = 0; i < H; i++) {
            temp[i] = temp[i] + 0.3f * chat_output[i];
        }
        
        memcpy(layer_output, temp, H * sizeof(float));
        free(temp);
        free(chat_output);
    }
    
    // Store final hidden state
    memcpy(net->hidden_state, layer_output, H * sizeof(float));
    
    // 5. Decode to output logits
    memset(net->output_logits, 0, net->vocab_size * sizeof(float));
    
    // Simple linear decode (can be enhanced)
    for (int i = 0; i < net->vocab_size; i++) {
        for (int j = 0; j < H; j++) {
            net->output_logits[i] += layer_output[j] * net->semantic_embedding[i * net->embed_dim + (j % net->embed_dim)];
        }
    }
    
    // Softmax for probabilities
    if (output_probs) {
        for (int i = 0; i < net->vocab_size; i++) {
            output_probs[i] = softmax_stable(net->output_logits, net->vocab_size, i);
        }
    }
    
    free(thought_embedding);
    free(attention_output);
    free(layer_output);
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

EXPORT void text_native_ssmn_reset(TextNativeSSMN *net) {
    memset(net->hidden_state, 0, net->hidden_dim * sizeof(float));
    memset(net->prev_hidden_state, 0, net->hidden_dim * sizeof(float));
    memset(net->internal_chat_state, 0, net->hidden_dim * sizeof(float));
    memset(net->window_buffer, 0, net->window_size * net->hidden_dim * sizeof(float));
    net->window_pos = 0;
    net->window_fill = 0;
}

EXPORT void text_native_ssmn_reset_synaptic_memory(TextNativeSSMN *net) {
    memset(net->synaptic_weights, 0, net->hidden_dim * net->hidden_dim * sizeof(float));
}

// ============================================================================
// GETTERS
// ============================================================================

EXPORT void text_native_ssmn_get_hidden_state(TextNativeSSMN *net, float *output) {
    memcpy(output, net->hidden_state, net->hidden_dim * sizeof(float));
}

EXPORT float text_native_ssmn_get_avg_importance(TextNativeSSMN *net) {
    return net->avg_importance;
}

EXPORT float text_native_ssmn_get_avg_synaptic_energy(TextNativeSSMN *net) {
    return net->avg_synaptic_energy;
}

EXPORT float text_native_ssmn_get_avg_attention_entropy(TextNativeSSMN *net) {
    return net->avg_attention_entropy;
}

EXPORT void text_native_ssmn_get_synaptic_weights(TextNativeSSMN *net, float *output) {
    memcpy(output, net->synaptic_weights, net->hidden_dim * net->hidden_dim * sizeof(float));
}

// ============================================================================
// INFO
// ============================================================================

EXPORT void text_native_ssmn_print_info(TextNativeSSMN *net) {
    printf("\n=== Text-Native SSMN ===\n");
    printf("Vocabulary Size: %d\n", net->vocab_size);
    printf("Embedding Dim: %d\n", net->embed_dim);
    printf("Hidden Dim: %d\n", net->hidden_dim);
    printf("Window Size: %d\n", net->window_size);
    printf("Static Layers: %d (80%%)\n", net->num_static_layers);
    printf("Plastic Layers: %d (20%%)\n", net->num_plastic_layers);
    printf("Plasticity η: %.4f\n", net->plasticity_eta);
    printf("Decay λ: %.4f\n", net->decay_lambda);
    printf("\n--- Statistics ---\n");
    printf("Avg Importance: %.4f\n", net->avg_importance);
    printf("Synaptic Energy: %.6f\n", net->avg_synaptic_energy);
    printf("Attention Entropy: %.4f\n", net->avg_attention_entropy);
    printf("Window Fill: %d/%d\n", net->window_fill, net->window_size);
}