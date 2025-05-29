document.addEventListener('DOMContentLoaded', () => {
    const modelCodeEl = document.getElementById('model-code');
    const analyzeButton = document.getElementById('analyze-button');
    const loadingIndicator = document.getElementById('loading-indicator');
    const networkGraphEl = document.getElementById('network-graph');
    const totalParamsEl = document.getElementById('total-params');
    const totalLayersEl = document.getElementById('total-layers');
    const trainableParamsEl = document.getElementById('trainable-params');

    const API_BASE_URL = 'http://localhost:8001/api/v1'; // 后端服务运行在8001端口

    let network = null; // Vis.js network instance

    // 测试后端连接
    async function testBackendConnection() {
        try {
            console.log('正在测试后端连接...');
            const response = await fetch(`${API_BASE_URL}/test`);
            if (response.ok) {
                const data = await response.json();
                console.log('后端连接测试成功:', data);
                return true;
            } else {
                console.error('后端连接测试失败:', response.status, response.statusText);
                return false;
            }
        } catch (error) {
            console.error('后端连接测试异常:', error);
            return false;
        }
    }

    // 页面加载后测试后端连接
    testBackendConnection().then(success => {
        if (!success) {
            console.warn('警告：无法连接到后端服务，请确保后端正在运行');
            alert('警告：无法连接到后端服务，请确保后端正在运行在 http://localhost:8001');
        } else {
            console.log('后端服务连接正常');
        }
    });

    analyzeButton.addEventListener('click', async () => {
        const modelCode = modelCodeEl.value.trim();
        if (!modelCode) {
            alert('请输入模型代码！');
            return;
        }

        // 清空旧的图表和统计信息
        if (network) {
            network.destroy();
            network = null;
        }
        networkGraphEl.innerHTML = ''; // 清空旧图
        totalParamsEl.textContent = '-';
        totalLayersEl.textContent = '-';
        trainableParamsEl.textContent = '-';
        loadingIndicator.style.display = 'block';
        analyzeButton.disabled = true;

        try {
            console.log('开始分析模型...');
            
            // 1. 提交分析任务
            console.log('提交分析请求到:', `${API_BASE_URL}/models/analyze`);
            const analyzeResponse = await fetch(`${API_BASE_URL}/models/analyze`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    code: modelCode,
                    model_name: "UserSubmittedModel",
                    analysis_options: {} // 使用默认分析选项
                })
            });

            console.log('分析请求响应状态:', analyzeResponse.status);

            if (!analyzeResponse.ok) {
                const errorText = await analyzeResponse.text();
                console.error('分析请求失败，响应内容:', errorText);
                throw new Error(`分析请求失败 (${analyzeResponse.status}): ${errorText}`);
            }

            const analyzeResult = await analyzeResponse.json();
            console.log('分析请求成功，任务ID:', analyzeResult.task_id);
            const taskId = analyzeResult.task_id;

            if (!taskId) {
                throw new Error('未能获取任务ID');
            }

            // 2. 轮询任务状态
            let taskStatus = '';
            let attempts = 0;
            const maxAttempts = 30; // 最多轮询30次 (约1分钟，如果间隔2秒)
            const pollInterval = 2000; // 2秒轮询一次

            while (taskStatus !== 'completed' && attempts < maxAttempts) {
                attempts++;
                await new Promise(resolve => setTimeout(resolve, pollInterval));

                console.log(`轮询任务状态，第 ${attempts} 次...`);
                const statusResponse = await fetch(`${API_BASE_URL}/models/${taskId}/status`);
                if (!statusResponse.ok) {
                    console.error('状态查询失败:', statusResponse.status, statusResponse.statusText);
                    throw new Error(`任务状态查询失败: ${statusResponse.statusText}`);
                }
                const statusResult = await statusResponse.json();
                taskStatus = statusResult.status;
                console.log('任务状态:', taskStatus);

                if (taskStatus === 'failed') {
                    throw new Error(`模型分析任务失败: ${statusResult.error || '未知错误'}`);
                }
                loadingIndicator.textContent = `正在分析中... (${statusResult.progress || 0}%) ${statusResult.current_stage || ''}`;
            }

            if (taskStatus !== 'completed') {
                throw new Error('分析任务超时或未成功完成。');
            }

            console.log('任务完成，获取可视化数据...');
            // 3. 获取可视化数据
            const vizResponse = await fetch(`${API_BASE_URL}/models/${taskId}/visualization`);
            if (!vizResponse.ok) {
                const errorText = await vizResponse.text();
                console.error('获取可视化数据失败:', errorText);
                throw new Error(`获取可视化数据失败 (${vizResponse.status}): ${errorText}`);
            }
            const vizData = await vizResponse.json();
            console.log('可视化数据获取成功:', vizData);

            // 4. 渲染网络图
            renderNetworkGraph(vizData.nodes, vizData.edges);

            // 5. 显示统计信息
            if (vizData.metadata && vizData.metadata.complexity) {
                const complexity = vizData.metadata.complexity;
                totalParamsEl.textContent = complexity.total_parameters?.toLocaleString() || '-';
                totalLayersEl.textContent = complexity.total_layers?.toLocaleString() || '-';
                trainableParamsEl.textContent = complexity.trainable_parameters?.toLocaleString() || '-';
            } else if (vizData.statistics) { // 备用，如果后端直接在visualization接口返回统计
                 totalParamsEl.textContent = vizData.statistics.total_parameters?.toLocaleString() || '-';
                 totalLayersEl.textContent = vizData.statistics.node_count?.toLocaleString() || '-'; // 用节点数近似层数
                 trainableParamsEl.textContent = vizData.statistics.trainable_parameters?.toLocaleString() || '-';
            }

            console.log('分析完成！');

        } catch (error) {
            console.error('分析过程中发生错误:', error);
            alert(`发生错误: ${error.message}`);
        } finally {
            loadingIndicator.style.display = 'none';
            loadingIndicator.textContent = '正在分析中，请稍候...'; // 重置提示文本
            analyzeButton.disabled = false;
        }
    });

    function renderNetworkGraph(nodesData, edgesData) {
        if (!nodesData || !edgesData) {
            networkGraphEl.innerHTML = '<p style="text-align:center; padding-top: 20px;">无法加载模型图数据。</p>';
            return;
        }

        console.log('渲染网络图，节点数:', nodesData.length, '边数:', edgesData.length);

        // 处理节点数据，改善形状显示
        const visNodes = nodesData.map(node => ({
            id: node.id,
            label: `${node.label || node.type}\n(${node.id})`,
            title: `类型: ${node.type}<br>参数: ${node.num_parameters?.toLocaleString() || 'N/A'}<br>输出形状: ${formatShape(node.output_shape)}`,
            shape: 'box',
            color: getNodeColor(node.type),
            font: { size: 11, face: 'arial' }
        }));

        // 处理边数据，减少标签重叠
        const visEdges = edgesData.map((edge, index) => ({
            from: edge.source,
            to: edge.target,
            arrows: 'to',
            label: getEdgeLabel(edge.connection_type),
            id: `edge_${index}`,
            font: { 
                size: 8, 
                color: '#666',
                strokeWidth: 2,
                strokeColor: 'white'
            },
            smooth: {
                type: 'cubicBezier',
                forceDirection: 'vertical',
                roundness: 0.2 + (index * 0.1) % 0.4  // 不同弯曲度避免重叠
            }
        }));

        // 手动添加缺失的连接（特别是conv3到fc1）
        const enhancedEdges = enhanceConnections(visNodes, visEdges);

        const data = {
            nodes: new vis.DataSet(visNodes),
            edges: new vis.DataSet(enhancedEdges),
        };

        const options = {
            layout: {
                hierarchical: {
                    direction: "UD",
                    sortMethod: "directed",
                    levelSeparation: 120,
                    nodeSpacing: 80,
                    treeSpacing: 100
                }
            },
            physics: {
                enabled: false
            },
            interaction: {
                dragNodes: true,
                dragView: true,
                zoomView: true,
                hover: true
            },
            nodes: {
                font: {
                    size: 11,
                    face: 'arial',
                    color: '#333'
                },
                margin: 8,
                widthConstraint: { maximum: 140 },
                heightConstraint: { minimum: 35 }
            },
            edges: {
                font: {
                    size: 8,
                    align: 'middle',
                    color: '#666',
                    strokeWidth: 2,
                    strokeColor: 'white'
                },
                color: {
                    color: '#848484',
                    highlight: '#5cb85c',
                    hover: '#5cb85c'
                },
                width: 2,
                smooth: {
                    type: 'cubicBezier',
                    forceDirection: 'vertical',
                    roundness: 0.3
                }
            }
        };

        network = new vis.Network(networkGraphEl, data, options);
        console.log('网络图渲染完成');
    }

    function formatShape(shape) {
        if (!shape || shape === 'N/A') return 'Dynamic';
        if (Array.isArray(shape)) {
            return shape.join(' × ');
        }
        return String(shape);
    }

    function getEdgeLabel(connectionType) {
        if (!connectionType) return '';
        // 简化标签，避免重叠
        const labelMap = {
            'sequential': '',  // 不显示标签，用箭头表示
            'reshape': 'view',
            'residual': '+',
            'attention': 'attn',
            'branch': 'split',
            'merge': 'cat'
        };
        return labelMap[connectionType] || connectionType;
    }

    function enhanceConnections(nodes, edges) {
        const enhancedEdges = [...edges];
        
        // 查找最后的卷积层和第一个线性层
        const convNodes = nodes.filter(node => 
            node.id.includes('conv') || 
            node.label.toLowerCase().includes('conv')
        );
        const linearNodes = nodes.filter(node => 
            node.id.includes('fc') || 
            node.label.toLowerCase().includes('linear')
        );
        
        if (convNodes.length > 0 && linearNodes.length > 0) {
            const lastConv = convNodes[convNodes.length - 1];
            const firstLinear = linearNodes[0];
            
            // 检查是否已存在连接
            const connectionExists = edges.some(edge => 
                edge.from === lastConv.id && edge.to === firstLinear.id
            );
            
            if (!connectionExists) {
                enhancedEdges.push({
                    from: lastConv.id,
                    to: firstLinear.id,
                    arrows: 'to',
                    label: 'flatten',
                    id: 'manual_flatten',
                    font: { 
                        size: 8, 
                        color: '#999',
                        strokeWidth: 2,
                        strokeColor: 'white'
                    },
                    color: { color: '#999' },
                    dashes: [5, 5],  // 虚线表示隐式连接
                    smooth: {
                        type: 'cubicBezier',
                        forceDirection: 'vertical',
                        roundness: 0.5
                    }
                });
            }
        }
        
        return enhancedEdges;
    }

    function getNodeColor(nodeType) {
        const typeLower = nodeType.toLowerCase();
        if (typeLower.includes('conv')) return { background: '#FFC107', border: '#FF8F00' };
        if (typeLower.includes('pool')) return { background: '#2196F3', border: '#1976D2' };
        if (typeLower.includes('linear') || typeLower.includes('fc')) return { background: '#4CAF50', border: '#388E3C' };
        if (typeLower.includes('relu') || typeLower.includes('activation')) return { background: '#E91E63', border: '#C2185B' };
        if (typeLower.includes('dropout')) return { background: '#9C27B0', border: '#7B1FA2' };
        if (typeLower.includes('batchnorm')) return { background: '#00BCD4', border: '#0097A7' };
        if (typeLower.includes('input')) return { background: '#FF5722', border: '#D84315' };
        if (typeLower.includes('output')) return { background: '#607D8B', border: '#455A64' };
        return { background: '#9E9E9E', border: '#616161' };
    }

});
