/**
 * NetView 可视化引擎
 * 负责网络结构图的渲染和交互
 */

class VisualizationEngine {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.svg = null;
        this.width = 0;
        this.height = 0;
        this.nodes = [];
        this.edges = [];
        this.simulation = null;
        
        // 颜色配置
        this.layerColors = {
            'Conv2d': '#FF6B6B',
            'Linear': '#4ECDC4',
            'MaxPool2d': '#45B7D1',
            'AvgPool2d': '#45B7D1',
            'BatchNorm2d': '#96CEB4',
            'BatchNorm1d': '#96CEB4',
            'ReLU': '#FFEAA7',
            'Sigmoid': '#FFEAA7',
            'Tanh': '#FFEAA7',
            'Dropout': '#DDA0DD',
            'default': '#95A5A6'
        };
        
        this.init();
    }

    /**
     * 初始化可视化环境
     */
    init() {
        // 清空容器
        this.container.innerHTML = '';
        
        // 获取容器尺寸
        const rect = this.container.getBoundingClientRect();
        this.width = rect.width || 800;
        this.height = rect.height || 600;
        
        // 创建SVG元素
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height)
            .style('background-color', '#fafafa');
            
        // 添加缩放和拖拽功能
        const zoom = d3.zoom()
            .scaleExtent([0.1, 3])
            .on('zoom', (event) => {
                this.svg.select('.graph-group')
                    .attr('transform', event.transform);
            });
            
        this.svg.call(zoom);
        
        // 创建图形组
        this.graphGroup = this.svg.append('g')
            .attr('class', 'graph-group');
            
        // 创建箭头标记
        this.createArrowMarkers();
        
        console.log('可视化引擎初始化完成');
    }

    /**
     * 创建箭头标记
     */
    createArrowMarkers() {
        const defs = this.svg.append('defs');
        
        defs.append('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 25)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#999');
    }

    /**
     * 渲染网络图
     * @param {Object} graphData - 图形数据
     */
    render(graphData) {
        if (!graphData || !graphData.nodes || !graphData.edges) {
            console.error('无效的图形数据');
            return;
        }

        console.log('开始渲染网络图', graphData);
        
        this.nodes = graphData.nodes;
        this.edges = graphData.edges;
        
        // 清除之前的内容
        this.graphGroup.selectAll('*').remove();
        
        // 创建力导向布局
        this.createForceSimulation();
        
        // 渲染边
        this.renderEdges();
        
        // 渲染节点
        this.renderNodes();
        
        // 启动模拟
        this.simulation.nodes(this.nodes);
        this.simulation.force('link').links(this.edges);
        this.simulation.alpha(1).restart();
        
        // 更新容器状态
        this.container.classList.add('has-content');
        
        console.log('网络图渲染完成');
    }

    /**
     * 创建力导向布局
     */
    createForceSimulation() {
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink()
                .id(d => d.id)
                .distance(100)
                .strength(0.5)
            )
            .force('charge', d3.forceManyBody()
                .strength(-300)
            )
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide()
                .radius(d => this.getNodeSize(d) + 10)
            );
    }

    /**
     * 渲染边
     */
    renderEdges() {
        this.edgeElements = this.graphGroup
            .selectAll('.edge')
            .data(this.edges)
            .enter()
            .append('line')
            .attr('class', 'edge')
            .attr('stroke', '#999')
            .attr('stroke-width', 2)
            .attr('marker-end', 'url(#arrow)')
            .style('opacity', 0.6);

        // 添加边的动画效果
        this.simulation.on('tick', () => {
            this.edgeElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
        });
    }

    /**
     * 渲染节点
     */
    renderNodes() {
        // 创建节点组
        this.nodeElements = this.graphGroup
            .selectAll('.node')
            .data(this.nodes)
            .enter()
            .append('g')
            .attr('class', 'node')
            .style('cursor', 'pointer')
            .call(this.createDragBehavior());

        // 添加节点圆形
        this.nodeElements
            .append('circle')
            .attr('r', d => this.getNodeSize(d))
            .attr('fill', d => this.getNodeColor(d))
            .attr('stroke', '#fff')
            .attr('stroke-width', 3)
            .style('filter', 'drop-shadow(2px 2px 4px rgba(0,0,0,0.1))');

        // 添加节点标签
        this.nodeElements
            .append('text')
            .text(d => this.getNodeLabel(d))
            .attr('text-anchor', 'middle')
            .attr('dy', d => this.getNodeSize(d) + 15)
            .attr('font-size', '12px')
            .attr('font-weight', 'bold')
            .attr('fill', '#333');

        // 添加参数信息
        this.nodeElements
            .append('text')
            .text(d => this.getNodeParams(d))
            .attr('text-anchor', 'middle')
            .attr('dy', d => this.getNodeSize(d) + 28)
            .attr('font-size', '10px')
            .attr('fill', '#666');

        // 添加交互事件
        this.addNodeInteractions();

        // 更新节点位置
        this.simulation.on('tick', () => {
            this.nodeElements
                .attr('transform', d => `translate(${d.x}, ${d.y})`);
        });
    }

    /**
     * 创建拖拽行为
     */
    createDragBehavior() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }

    /**
     * 添加节点交互事件
     */
    addNodeInteractions() {
        this.nodeElements
            .on('mouseover', (event, d) => {
                this.showTooltip(event, d);
                this.highlightNode(d);
            })
            .on('mouseout', (event, d) => {
                this.hideTooltip();
                this.unhighlightNode(d);
            })
            .on('click', (event, d) => {
                this.onNodeClick(d);
            });
    }

    /**
     * 获取节点大小
     */
    getNodeSize(node) {
        const paramCount = node.params ? Object.keys(node.params).length : 0;
        return Math.max(20, Math.min(50, 20 + paramCount * 2));
    }

    /**
     * 获取节点颜色
     */
    getNodeColor(node) {
        return this.layerColors[node.type] || this.layerColors.default;
    }

    /**
     * 获取节点标签
     */
    getNodeLabel(node) {
        return node.label || node.type || node.id;
    }

    /**
     * 获取节点参数信息
     */
    getNodeParams(node) {
        if (!node.params) return '';
        const paramCount = Object.keys(node.params).length;
        return paramCount > 0 ? `${paramCount} 参数` : '';
    }

    /**
     * 显示工具提示
     */
    showTooltip(event, node) {
        const tooltip = d3.select('body')
            .append('div')
            .attr('class', 'tooltip')
            .style('position', 'absolute')
            .style('background', 'rgba(0,0,0,0.8)')
            .style('color', 'white')
            .style('padding', '8px')
            .style('border-radius', '4px')
            .style('font-size', '12px')
            .style('pointer-events', 'none')
            .style('z-index', '1000');

        let content = `<strong>${node.label || node.type}</strong><br/>`;
        if (node.params) {
            content += Object.entries(node.params)
                .map(([key, value]) => `${key}: ${value}`)
                .join('<br/>');
        }
        if (node.shape) {
            content += `<br/>形状: ${JSON.stringify(node.shape)}`;
        }

        tooltip.html(content)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px');
    }

    /**
     * 隐藏工具提示
     */
    hideTooltip() {
        d3.selectAll('.tooltip').remove();
    }

    /**
     * 高亮节点
     */
    highlightNode(node) {
        this.nodeElements.selectAll('circle')
            .style('opacity', d => d === node ? 1 : 0.3);
        this.edgeElements
            .style('opacity', d => 
                d.source === node || d.target === node ? 1 : 0.1
            );
    }

    /**
     * 取消高亮
     */
    unhighlightNode(node) {
        this.nodeElements.selectAll('circle')
            .style('opacity', 1);
        this.edgeElements
            .style('opacity', 0.6);
    }

    /**
     * 节点点击事件
     */
    onNodeClick(node) {
        console.log('节点被点击:', node);
        // 可以在这里添加更多交互逻辑
    }

    /**
     * 导出SVG
     */
    exportSVG() {
        const svgElement = this.container.querySelector('svg');
        const serializer = new XMLSerializer();
        const svgString = serializer.serializeToString(svgElement);
        const blob = new Blob([svgString], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = 'network_graph.svg';
        link.click();
        
        URL.revokeObjectURL(url);
    }

    /**
     * 重置视图
     */
    resetView() {
        this.svg.transition()
            .duration(750)
            .call(
                d3.zoom().transform,
                d3.zoomIdentity
            );
    }

    /**
     * 销毁可视化实例
     */
    destroy() {
        if (this.simulation) {
            this.simulation.stop();
        }
        if (this.container) {
            this.container.innerHTML = '';
        }
    }
}

// 创建全局可视化引擎实例
window.visualizationEngine = null;

// 初始化可视化引擎
function initVisualization() {
    if (window.visualizationEngine) {
        window.visualizationEngine.destroy();
    }
    window.visualizationEngine = new VisualizationEngine('graph-container');
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 检查D3.js是否加载
    if (typeof d3 === 'undefined') {
        console.warn('D3.js未加载，可视化功能将不可用');
        return;
    }
    
    initVisualization();
});

// 导出类
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { VisualizationEngine };
}
