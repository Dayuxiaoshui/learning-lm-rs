use std::sync::Arc;
use crate::{kvcache::KVCache, model::Llama};

/// Chat 结构体，存储对话的相关信息，包括对话ID、消息、输入ID、缓存等。
pub struct Chat<'a> {
    id: usize,
    message: String,
    his_message: String,
    input_ids: Vec<u32>,
    re_input_ids: Vec<u32>,
    model: &'a Llama<f32>,
    cache: Arc<KVCache<f32>>,
    h_cache: Arc<KVCache<f32>>,
}

impl<'a> Chat<'a> {
    /// 创建一个新的 Chat 实例。
    pub fn new_chat(id: usize, model: &'a Llama<f32>) -> Chat<'a> {
        Chat {
            id,
            message: String::new(),
            his_message: String::new(),
            input_ids: Vec::new(),
            re_input_ids: Vec::new(),
            cache: Arc::new(model.new_cache()),  // 初始化缓存
            h_cache: Arc::new(model.new_cache()), // 初始化历史缓存
            model,
        }
    }

    /// 根据输入生成响应，使用 Llama 模型生成对话。
    pub fn generate(
        model: &'a Llama<f32>,
        message: String,
        input: &[u32],
        cache: Arc<KVCache<f32>>,
    ) -> (Vec<u32>, KVCache<f32>) {
        // 检查 缓存是否可以被获取
        let kv_cache = Arc::into_inner(cache);

        // 如果消息为空，创建新对话，否则使用现有缓存
        if message.is_empty() {
            model.generate_chat(input, 100, 0.9, 4, 1., None)
        } else {
            model.generate_chat(input, 100, 0.9, 4, 1., kv_cache)
        }
    }

    /// 获取当前对话的消息。
    pub fn get_message(&self) -> &String {
        &self.message
    }

    /// 获取当前输入的ID。
    pub fn get_ids(&self) -> Vec<u32> {
        self.input_ids.clone()
    }

    /// 获取重新输入的ID。
    pub fn get_re_ids(&self) -> Vec<u32> {
        self.re_input_ids.clone()
    }

    /// 获取历史消息。
    pub fn get_his_message(&self) -> &String {
        &self.his_message
    }

    /// 获取当前缓存。
    pub fn get_cache(&self) -> Arc<KVCache<f32>> {
        Arc::clone(&self.cache)
    }

    /// 获取历史缓存。
    pub fn get_h_cache(&self) -> Arc<KVCache<f32>> {
        Arc::clone(&self.h_cache)
    }

    /// 设置历史缓存。
    pub fn set_his_cache(&mut self, his_cache: Arc<KVCache<f32>>) {
        self.h_cache = his_cache;
    }

    /// 设置当前缓存。
    pub fn set_cache(&mut self, cache: Arc<KVCache<f32>>) {
        self.cache = cache;
    }

    /// 设置当前消息。
    pub fn set_message(&mut self, message: String) {
        self.message = message;
    }

    /// 设置输入的ID。
    pub fn set_input_ids(&mut self, input_ids: Vec<u32>) {
        self.input_ids = input_ids;
    }

    /// 设置重新输入的ID。
    pub fn set_re_input_ids(&mut self, re_input_ids: Vec<u32>) {
        self.re_input_ids = re_input_ids;
    }

    /// 设置历史消息。
    pub fn set_his_message(&mut self, his_message: String) {
        self.his_message = his_message;
    }
}
