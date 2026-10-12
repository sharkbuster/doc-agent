/* ============================================================
 * 拾光优选 · 售后客服智能体 —— 前端交互
 *
 * 纯原生 JS，无框架、无构建、无外部资源。
 * 白箱核心：把手动解析的 SSE 事件逐帧渲染成「状态条 + 依据条款 + 工具调用」。
 * ============================================================ */
(function () {
  'use strict';

  /* ---------------------------------------------------------
   * 0. 元素引用
   *    这些 id 全部在 index.html 里真实存在（见交付说明的对照表）。
   * ------------------------------------------------------- */
  var elMessages     = document.getElementById('messages');
  var elChatSub      = document.getElementById('chat-sub');
  var elHealthBody   = document.getElementById('health-body');
  var elToolsToggle  = document.getElementById('tools-toggle');
  var elToolsBody    = document.getElementById('tools-body');
  var elSidebarAlert = document.getElementById('sidebar-alert');
  var elStreamAlert  = document.getElementById('stream-alert');
  var elComposer     = document.getElementById('composer');
  var elInput        = document.getElementById('input');
  var elBtnSend      = document.getElementById('btn-send');
  var elBtnStop      = document.getElementById('btn-stop');
  var elBtnReset     = document.getElementById('btn-reset');

  /* ---------------------------------------------------------
   * 1. 小工具
   * ------------------------------------------------------- */
  function $(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function clear(node) {
    while (node && node.firstChild) { node.removeChild(node.firstChild); }
  }

  function fmtScore(value) {
    var n = Number(value);
    return isFinite(n) ? n.toFixed(2) : '—';
  }

  function prettyJson(value) {
    try {
      return JSON.stringify(value, null, 2);
    } catch (err) {
      return String(value);
    }
  }

  /* ---------------------------------------------------------
   * 2. 全局提示条（接口挂了也绝不白屏）
   * ------------------------------------------------------- */
  function showAlert(target, kind, message) {
    if (!target) { return; }
    target.className = 'alert alert-' + kind;
    clear(target);
    target.appendChild($('span', null, message));
    target.hidden = false;
  }

  function hideAlert(target) {
    if (!target) { return; }
    target.hidden = true;
  }

  /* ---------------------------------------------------------
   * 3. 启动引导：health / tools / demo orders
   *    三者互相独立，任何一个失败都只影响自己的那一块。
   * ------------------------------------------------------- */

  function renderHealth(data) {
    clear(elHealthBody);
    if (!data || typeof data !== 'object') {
      elHealthBody.appendChild($('p', 'muted', '状态信息格式异常。'));
      return;
    }

    var llm = data.llm || {};
    var corpus = data.corpus || {};
    var retrieval = data.retrieval || {};
    var provider = llm.provider || '未知';
    var model = llm.model || '未知';
    var isStub = String(provider).toLowerCase() === 'stub' && !llm.api_key_configured;

    function kv(key, value, mono) {
      var row = $('div', 'kv');
      row.appendChild($('span', 'kv-k', key));
      row.appendChild($('span', 'kv-v' + (mono ? ' mono' : ''), value));
      elHealthBody.appendChild(row);
    }

    kv('模型', provider + ' / ' + model, true);
    kv('知识库', (corpus.documents != null ? corpus.documents : '?') + ' 篇文档 · ' +
                 (corpus.chunks != null ? corpus.chunks : '?') + ' 个片段');
    kv('向量后端', retrieval.backend || data.embedding || '—', true);
    kv('召回条数', retrieval.top_k != null ? 'top_k = ' + retrieval.top_k : '—');
    kv('拒答阈值', retrieval.refusal_threshold != null ? retrieval.refusal_threshold : '—');
    if (data.version) { kv('接口版本', data.version, true); }

    if (isStub) {
      var warn = $('div', 'alert alert-warn');
      warn.style.marginTop = '10px';
      var strong = $('strong', null, '离线桩模型');
      warn.appendChild(strong);
      warn.appendChild($('span', null, '：未配置真实模型密钥，回答仅供参考。'));
      elHealthBody.appendChild(warn);
    }
  }

  function renderTools(list) {
    clear(elToolsBody);
    if (!Array.isArray(list) || !list.length) {
      elToolsBody.appendChild($('p', 'muted', '暂时读不到工具清单。'));
      return;
    }
    list.forEach(function (tool) {
      var item = $('div', 'tool-item');
      item.appendChild($('div', 'tool-name', tool && tool.name ? tool.name : '(未命名工具)'));
      if (tool && tool.description) {
        item.appendChild($('div', 'tool-desc', tool.description));
      }
      var params = tool && tool.parameters;
      if (params && typeof params === 'object') {
        var keys = Object.keys(params);
        if (keys.length) {
          var lines = keys.map(function (k) { return k + '：' + params[k]; });
          item.appendChild($('div', 'tool-params', lines.join('　·　')));
        }
      }
      elToolsBody.appendChild(item);
    });
  }

  function toggleTools() {
    var open = elToolsToggle.getAttribute('aria-expanded') === 'true';
    elToolsToggle.setAttribute('aria-expanded', open ? 'false' : 'true');
    elToolsBody.hidden = open;
  }

  /* ---------- 欢迎卡片与示例问题 ---------- */

  var POLICY_SAMPLES = [
    '7 天无理由退货的期限是几天？',
    '退货运费谁承担？',
    '发票可以开电子版吗？'
  ];

  var OUT_OF_SCOPE_SAMPLES = [
    '怎么申请提高信用卡额度？'
  ];

  function buildSamples(orders) {
    /* 三类问题都要覆盖：
     *   1) 纯政策类 —— 固定文案
     *   2) 订单数据类 —— 用 /api/demo/orders 的真实订单号拼
     *   3) 超纲类 —— 知识库和工具都答不了
     */
    var groups = [];

    groups.push({ title: '政策咨询', items: POLICY_SAMPLES.slice(0, 2) });

    var orderItems = [];
    if (Array.isArray(orders)) {
      orders.slice(0, 3).forEach(function (order) {
        if (!order || !order.order_id) { return; }
        var label = order.order_id;
        if (order.status_label) { label += '（' + order.status_label + '）'; }
        orderItems.push({
          text: order.order_id + ' 无理由退货能退多少钱？',
          label: label
        });
      });
    }
    if (orderItems.length) {
      groups.push({ title: '订单数据', items: orderItems });
    } else {
      /* 订单接口挂了也要给出可点的示例，只是退化成占位订单号 */
      groups.push({
        title: '订单数据',
        items: [{ text: 'SO20261001001 无理由退货能退多少钱？', label: '订单号示例' }]
      });
    }

    groups.push({ title: '其他', items: OUT_OF_SCOPE_SAMPLES });
    return groups;
  }

  function renderWelcome(orders, orderLoadFailed) {
    var card = $('div', 'welcome');
    card.appendChild($('h3', null, '你好，我是拾光优选的售后客服助手。'));
    card.appendChild($('p', null,
      '我能查平台的售后政策，也能查你的订单、物流、库存和退款金额。' +
      '每一次回答都会公开「依据的条款」和「工具调用过程」，' +
      '没有依据时我会直接说查不到，不会编。'));

    buildSamples(orders).forEach(function (group) {
      card.appendChild($('span', 'chip-cat', group.title));
      var wrap = $('div', 'chips');
      group.items.forEach(function (item) {
        var text = typeof item === 'string' ? item : item.text;
        var label = typeof item === 'string' ? item : (item.label || item.text);
        var chip = $('button', 'chip', label);
        chip.type = 'button';
        chip.title = text.length > 40 ? text : '';
        chip.dataset.q = text;
        wrap.appendChild(chip);
      });
      card.appendChild(wrap);
    });

    if (orderLoadFailed) {
      var note = $('div', 'alert alert-warn');
      note.style.marginTop = '14px';
      note.appendChild($('span', null,
        '订单示例接口暂时不可用，上面用的是占位订单号，实际提问时请填写你的真实订单号。'));
      card.appendChild(note);
    }

    elMessages.appendChild(card);
  }

  function removeWelcome() {
    var card = elMessages.querySelector('.welcome');
    if (card) { card.remove(); }
  }

  /* ---------------------------------------------------------
   * 4. 消息渲染
   * ------------------------------------------------------- */

  var history = [];      /* [{role, content}] —— 只放已完成的轮次 */
  var streaming = false;
  var activeController = null;

  function scrollToBottom() {
    elMessages.scrollTop = elMessages.scrollHeight;
  }

  function appendUserMessage(text) {
    var wrap = $('div', 'msg msg-user');
    var bubble = $('div', 'bubble', text);
    wrap.appendChild(bubble);
    elMessages.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  /* 一条助手消息的全部活动部件 */
  function buildAssistantMessage() {
    var wrap = $('div', 'msg msg-assistant');
    var col = $('div', 'col');

    var status = $('div', 'status-line');
    var bubble = $('div', 'assistant-bubble');
    var textNode = $('div', 'answer-text');
    bubble.appendChild(textNode);

    var folds = $('div', 'folds');
    folds.style.display = 'flex';
    folds.style.flexDirection = 'column';
    folds.style.gap = '8px';

    col.appendChild(status);
    col.appendChild(bubble);
    col.appendChild(folds);
    wrap.appendChild(col);
    elMessages.appendChild(wrap);

    /* 给这条助手消息一个稳定 uid：引用标记靠它定位到本轮的条款区块 */
    var uid = ++uidSeq;
    wrap.dataset.uid = String(uid);

    var ctx = {
      uid: uid,
      wrap: wrap,
      status: status,
      bubble: bubble,
      textNode: textNode,
      folds: folds,
      rawText: '',
      steps: [],
      citations: [],
      toolCalls: [],
      hitCount: 0,
      topDense: null,
      kbMatched: null,
      droppedCitations: [],
      pendingCalls: {}
    };
    /* 状态条与文本之间留一点距离 */
    status.style.marginBottom = '0';
    scrollToBottom();
    return ctx;
  }

  /* ---------- 状态条 ---------- */

  function addStep(ctx, label, kind) {
    var step = $('span', 'step' + (kind ? ' is-' + kind : ''));
    step.appendChild($('span', 'step-dot'));
    step.appendChild($('span', null, label));
    ctx.status.appendChild(step);
    ctx.steps.push(step);
    scrollToBottom();
    return step;
  }

  function settleStep(step, kind) {
    if (!step) { return; }
    step.classList.remove('is-running');
    if (kind) { step.classList.add('is-' + kind); }
  }

  /* ---------- 折叠区块 ---------- */

  function createFold(title, countText) {
    var fold = $('div', 'fold');
    var head = $('button', 'fold-head');
    head.type = 'button';
    head.setAttribute('aria-expanded', 'false');
    head.appendChild($('span', null, title));
    if (countText) { head.appendChild($('span', 'fold-count', countText)); }
    head.appendChild($('span', 'chev'));

    var body = $('div', 'fold-body');
    body.hidden = true;                 /* 默认折叠；[hidden] 由 CSS 强制生效 */

    head.addEventListener('click', function () {
      var open = head.getAttribute('aria-expanded') === 'true';
      head.setAttribute('aria-expanded', open ? 'false' : 'true');
      body.hidden = open;
    });

    fold.appendChild(head);
    fold.appendChild(body);
    return { fold: fold, head: head, body: body, count: countText };
  }

  /* ---------- 内联引用标记 ---------- */
  /* 先整段转义，再把 [n] 换成可点击的小标记 —— 顺序不能反，
     否则会把用户/模型文本里的尖括号当成 HTML 执行。 */

  function renderAnswerText(ctx) {
    var known = {};
    ctx.citations.forEach(function (c) { known[String(c.index)] = true; });

    var esc = ctx.rawText
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    var html = esc.replace(/\[(\d+)\]/g, function (whole, digits) {
      if (!known[digits]) { return whole; }        /* 匹配不到就保持纯文本 */
      return '<button type="button" class="cite-mark" data-cite="' + digits +
             '" title="查看第 ' + digits + ' 条依据">[' + digits + ']</button>';
    });

    ctx.textNode.innerHTML = html;
  }

  function setAnswerText(ctx, text, streamingNow) {
    ctx.rawText = text;
    renderAnswerText(ctx);
    if (streamingNow) {
      var cursor = $('span', 'cursor');
      ctx.textNode.appendChild(cursor);
    }
  }

  /* ---------- 条款与工具区块（在 answer 帧后一次性建好） ---------- */

  function buildFolds(ctx, citations, toolCalls, hits) {
    var citeList = Array.isArray(citations) && citations.length
      ? citations
      : (Array.isArray(hits) ? hits : []);

    /* 依据的条款 */
    if (citeList.length) {
      var cit = createFold('依据的条款', citeList.length + ' 条');
      citeList.forEach(function (item) {
        var row = $('div', 'cite');
        row.id = 'cite-' + ctx.uid + '-' + item.index;

        var head = $('div', 'cite-head');
        head.appendChild($('span', 'cite-idx', '[' + item.index + ']'));
        head.appendChild($('span', 'cite-label', item.label || ('片段 ' + item.index)));
        var metaBits = [];
        if (item.dense != null) { metaBits.push('相似度 ' + fmtScore(item.dense)); }
        if (item.score != null) { metaBits.push('融合 ' + fmtScore(item.score)); }
        if (metaBits.length) {
          head.appendChild($('span', 'cite-meta', metaBits.join(' · ')));
        }
        row.appendChild(head);
        row.appendChild($('div', 'cite-text', item.text || '（无条款原文）'));
        cit.body.appendChild(row);
      });
      ctx.folds.appendChild(cit.fold);
      ctx.citeFold = cit;
    }

    /* 工具调用过程 */
    if (Array.isArray(toolCalls) && toolCalls.length) {
      var badCount = toolCalls.filter(function (c) { return c && c.ok === false; }).length;
      var toolFold = createFold(
        '工具调用过程',
        toolCalls.length + ' 次' + (badCount ? ' · ' + badCount + ' 次失败' : '')
      );
      ctx.callBody = toolFold.body;
      ctx.folds.appendChild(toolFold.fold);
      toolCalls.forEach(function (call) { appendToolCall(ctx, call); });
    }

    /* 引用被系统丢弃 —— 这是保护机制，要显眼 */
    if (Array.isArray(ctx.droppedCitations) && ctx.droppedCitations.length) {
      var warn = $('div', 'alert alert-warn');
      var strong = $('strong', null, '已拦截编造的引用');
      warn.appendChild(strong);
      warn.appendChild($('span', null,
        '：模型引用了不存在的条款编号 ' +
        ctx.droppedCitations.map(function (n) { return '[' + n + ']'; }).join('、') +
        '，系统已从回答中移除（可用条款共 ' + ctx.citationAvailable + ' 条）。' +
        '被移除的编号不会显示为可点击标记。'));
      ctx.folds.appendChild(warn);
    }
  }

  function appendToolCall(ctx, call) {
    if (!ctx.callBody) { return; }
    var data = call || {};
    var ok = data.ok !== false;

    var box = $('div', 'call' + (ok ? '' : ' is-bad'));
    var head = $('div', 'call-head');
    head.appendChild($('span', 'call-name', data.name || '(未知工具)'));
    head.appendChild($('span', 'call-badge', ok ? '成功' : '失败'));
    if (data.id) { head.appendChild($('span', 'call-sub', String(data.id))); }
    box.appendChild(head);

    var argPre = $('pre', null, prettyJson(data.args || {}));
    argPre.setAttribute('aria-label', '参数');
    box.appendChild(argPre);

    if (data.summary) {
      box.appendChild($('div', 'call-summary', data.summary));
    }
    ctx.callBody.appendChild(box);
  }

  /* ---------- 点击引用标记 → 展开并高亮对应条款 ---------- */

  function revealCitation(uid, index) {
    var row = document.getElementById('cite-' + uid + '-' + index);
    if (!row) { return false; }
    var body = row.closest ? row.closest('.fold-body') : null;
    if (body) {
      var head = body.parentNode.querySelector('.fold-head');
      body.hidden = false;
      if (head) { head.setAttribute('aria-expanded', 'true'); }
    }
    row.classList.add('is-flash');
    if (row.scrollIntoView) {
      row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
    window.setTimeout(function () { row.classList.remove('is-flash'); }, 1600);
    return true;
  }

  /* ---------------------------------------------------------
   * 5. SSE 解析
   *    不能用 EventSource（要 POST + JSON body），所以手动读流。
   *    分帧规则：事件之间以空行分隔；同一事件内多行 data: 用 \n 拼接。
   * ------------------------------------------------------- */

  function createSSEParser(onEvent, onRawError) {
    var buffer = '';

    function handleBlock(block) {
      var dataLines = [];
      block.split(/\r?\n/).forEach(function (line) {
        if (!line || line.charAt(0) === ':') { return; }   /* 空行 / 注释心跳 */
        var colon = line.indexOf(':');
        var field = colon === -1 ? line : line.slice(0, colon);
        var value = colon === -1 ? '' : line.slice(colon + 1);
        if (value.charAt(0) === ' ') { value = value.slice(1); }
        if (field === 'data') { dataLines.push(value); }
      });
      if (!dataLines.length) { return; }

      var payload = dataLines.join('\n');
      if (payload === '[DONE]') { return; }
      try {
        onEvent(JSON.parse(payload));
      } catch (err) {
        if (onRawError) { onRawError(payload, err); }
      }
    }

    return {
      push: function (text) {
        buffer += text;
        /* 只有拿到完整的分隔空行才切帧，避免把半个 JSON 当帧解析 */
        var match;
        var splitter = /\r?\n\r?\n/;
        while ((match = splitter.exec(buffer)) !== null) {
          var block = buffer.slice(0, match.index);
          buffer = buffer.slice(match.index + match[0].length);
          if (block.trim()) { handleBlock(block); }
        }
      },
      flush: function () {
        if (buffer.trim()) {
          handleBlock(buffer);
          buffer = '';
        }
      }
    };
  }

  /* ---------------------------------------------------------
   * 6. 一轮问答
   * ------------------------------------------------------- */

  var uidSeq = 0;

  function setBusy(busy) {
    streaming = busy;
    elInput.disabled = busy;
    elBtnSend.disabled = busy;
    elBtnSend.hidden = busy;
    elBtnStop.hidden = !busy;
    elBtnReset.disabled = busy;
    if (!busy) {
      elInput.focus();
    }
  }

  function dispatchEvent(ctx, ev) {
    var type = ev && ev.type;

    if (type === 'retrieval') {
      ctx.hitCount = Array.isArray(ev.hits) ? ev.hits.length : 0;
      ctx.topDense = ev.top_dense;
      ctx.kbMatched = ev.kb_matched;
      var label;
      if (ctx.hitCount > 0) {
        label = '检索到 ' + ctx.hitCount + ' 条条款（相似度 ' + fmtScore(ev.top_dense) + '）';
      } else {
        label = '知识库未命中相关条款（相似度 ' + fmtScore(ev.top_dense) + '）';
      }
      settleStep(addStep(ctx, label, ctx.hitCount > 0 ? null : 'warn'));
      return;
    }

    if (type === 'tool_call') {
      var step = addStep(ctx, '调用 ' + (ev.name || '工具') + '…');
      step.classList.add('is-running');
      ctx.pendingCalls = ctx.pendingCalls || {};
      ctx.pendingCalls[String(ev.id)] = step;
      return;
    }

    if (type === 'tool_result') {
      var key = String(ev.id);
      var pending = ctx.pendingCalls ? ctx.pendingCalls[key] : null;
      var ok = ev.ok !== false;
      if (pending) {
        pending.lastChild.textContent =
          (ev.name || '工具') + (ok ? ' 已完成' : ' 失败');
        settleStep(pending, ok ? 'ok' : 'warn');
      } else {
        settleStep(addStep(ctx, (ev.name || '工具') + (ok ? ' 已完成' : ' 失败'),
          ok ? 'ok' : 'warn'));
      }
      ctx.toolCalls.push(ev);
      if (ctx.callBody) { appendToolCall(ctx, ev); }
      return;
    }

    if (type === 'answer_delta') {
      if (!ctx.deltaStep) {
        ctx.deltaStep = addStep(ctx, '回答中', 'running');
        ctx.deltaStep.classList.add('is-running');
      }
      setAnswerText(ctx, ctx.rawText + (ev.text || ''), true);
      scrollToBottom();
      return;
    }

    if (type === 'citation_warning') {
      ctx.droppedCitations = Array.isArray(ev.dropped) ? ev.dropped : [];
      ctx.citationAvailable = ev.available;
      settleStep(addStep(ctx,
        '拦截编造引用 ' + ctx.droppedCitations.map(function (n) { return '[' + n + ']'; }).join('、'),
        'warn'));
      return;
    }

    if (type === 'answer') {
      ctx.completed = true;          /* 拿到 answer 帧 = 这一轮后端确实收尾了 */
      ctx.citations = Array.isArray(ev.citations) ? ev.citations : [];
      ctx.hitCount = Array.isArray(ev.hits) ? ev.hits.length : ctx.hitCount;
      setAnswerText(ctx, String(ev.answer != null ? ev.answer : ctx.rawText), false);

      if (ev.refused) {
        var tag = $('div', 'refused-tag', '本轮无依据，系统按规则拒答');
        ctx.bubble.insertBefore(tag, ctx.bubble.firstChild);
      }

      if (ctx.deltaStep) {
        settleStep(ctx.deltaStep, 'ok');
        ctx.deltaStep.lastChild.textContent =
          '回答完成' + (ev.steps != null ? '（' + ev.steps + ' 步）' : '');
      } else {
        settleStep(addStep(ctx, '直接回答', 'ok'));
      }
      if (ev.top_dense != null && ctx.topDense == null) { ctx.topDense = ev.top_dense; }

      buildFolds(ctx, ctx.citations, ev.tool_calls, ev.hits);
      return;
    }

    if (type === 'error') {
      showAlert(elStreamAlert, 'bad', '服务端返回错误：' + (ev.message || '未知错误'));
      settleStep(addStep(ctx, '服务端出错', 'warn'));
      return;
    }

    /* done 帧不需要额外处理：收尾逻辑在流结束处统一做 */
  }

  function finishTurn(ctx, note) {
    if (ctx.deltaStep) { settleStep(ctx.deltaStep, 'ok'); }
    if (!ctx.rawText && !note) {
      setAnswerText(ctx, '（本轮没有返回内容。）', false);
    }
    if (note) { showAlert(elStreamAlert, 'warn', note); }
    renderAnswerText(ctx);   /* 去掉流式光标 */
  }

  function send(question) {
    if (streaming) { return; }
    var text = String(question || '').trim();
    if (!text) { return; }

    removeWelcome();
    hideAlert(elStreamAlert);

    appendUserMessage(text);
    var ctx = buildAssistantMessage();

    var outgoingHistory = history.slice(-10);

    setBusy(true);
    activeController = ('AbortController' in window) ? new AbortController() : null;

    var parser = createSSEParser(
      function (ev) {
        try {
          dispatchEvent(ctx, ev);
        } catch (err) {
          showAlert(elStreamAlert, 'warn', '渲染某条流式事件时出错：' + err.message);
        }
      },
      function (raw, err) {
        showAlert(elStreamAlert, 'warn',
          '收到一帧无法解析的数据，已跳过：' + err.message);
      }
    );

    var requestInit = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({ question: text, history: outgoingHistory })
    };
    if (activeController) { requestInit.signal = activeController.signal; }

    fetch('/api/chat/stream', requestInit)
      .then(function (response) {
        if (!response.ok) {
          throw new Error('HTTP ' + response.status + ' ' + response.statusText);
        }
        if (!response.body || typeof response.body.getReader !== 'function') {
          throw new Error('当前浏览器不支持流式读取响应体');
        }
        var reader = response.body.getReader();
        var decoder = new TextDecoder('utf-8');

        function pump() {
          return reader.read().then(function (result) {
            if (result.done) {
              parser.flush();
              return null;
            }
            parser.push(decoder.decode(result.value, { stream: true }));
            return pump();
          });
        }
        return pump();
      })
      .then(function () {
        finishTurn(ctx, null);
        /* 只有后端正常收尾（answer 帧）的轮次才进历史，避免半截回答污染上下文 */
        if (ctx.completed) {
          history.push({ role: 'user', content: text });
          history.push({ role: 'assistant', content: ctx.rawText });
        }
      })
      .catch(function (err) {
        var aborted = err && err.name === 'AbortError';
        if (aborted) {
          finishTurn(ctx, null);
          settleStep(addStep(ctx, '已手动停止', 'warn'));
          if (ctx.completed) {
            history.push({ role: 'user', content: text });
            history.push({ role: 'assistant', content: ctx.rawText });
          }
        } else {
          var msg = '对话接口调用失败：' + (err && err.message ? err.message : String(err)) +
                    '。请确认后端已启动（GET /api/health），然后重试。';
          showAlert(elStreamAlert, 'bad', msg);
          if (!ctx.rawText) {
            ctx.bubble.appendChild($('div', 'muted', '没有取到回答。'));
          }
          settleStep(addStep(ctx, '请求失败', 'warn'));
        }
      })
      .then(function () {
        activeController = null;
        setBusy(false);
        scrollToBottom();
      });
  }

  /* ---------------------------------------------------------
   * 7. 事件绑定
   * ------------------------------------------------------- */

  elComposer.addEventListener('submit', function (e) {
    e.preventDefault();
    var value = elInput.value;
    elInput.value = '';
    autoGrow();
    send(value);
  });

  function autoGrow() {
    elInput.style.height = 'auto';
    elInput.style.height = Math.min(elInput.scrollHeight, 168) + 'px';
  }

  elInput.addEventListener('input', autoGrow);

  elInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!streaming) {
        var value = elInput.value;
        elInput.value = '';
        autoGrow();
        send(value);
      }
    }
  });

  elBtnStop.addEventListener('click', function () {
    if (activeController) { activeController.abort(); }
  });

  elBtnReset.addEventListener('click', function () {
    if (streaming && activeController) { activeController.abort(); }
    history = [];
    clear(elMessages);
    hideAlert(elStreamAlert);
    renderWelcome(cachedOrders, ordersFailed);
    elInput.focus();
  });

  elToolsToggle.addEventListener('click', toggleTools);
  elToolsToggle.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
      e.preventDefault();
      toggleTools();
    }
  });

  /* 引用标记：事件委托，边流边渲染也能点 */
  elMessages.addEventListener('click', function (e) {
    var target = e.target;
    if (!target || !target.closest) { return; }
    var mark = target.closest('.cite-mark');
    if (!mark) { return; }
    var index = mark.getAttribute('data-cite');
    var turn = mark.closest('.msg-assistant');
    var uid = turn && turn.dataset ? turn.dataset.uid : null;
    if (uid) { revealCitation(uid, index); }
  });

  /* ---------------------------------------------------------
   * 8. 启动
   * ------------------------------------------------------- */

  var cachedOrders = [];
  var ordersFailed = false;
  var booted = false;

  function boot() {
    if (booted) { return; }
    booted = true;

    /* health（失败不影响聊天） */
    fetch('/api/health')
      .then(function (r) {
        if (!r.ok) { throw new Error('HTTP ' + r.status); }
        return r.json();
      })
      .then(function (data) {
        renderHealth(data);
        if (data && data.retrieval && data.retrieval.top_k) {
          elChatSub.textContent =
            '退换货 · 退款 · 运费 · 物流 · 发票　|　召回 ' + data.retrieval.top_k + ' 条 / 拒答阈值 ' +
            data.retrieval.refusal_threshold;
        }
      })
      .catch(function (err) {
        clear(elHealthBody);
        elHealthBody.appendChild($('p', 'muted',
          '读不到系统状态：' + (err && err.message ? err.message : '接口不可用')));
        showAlert(elSidebarAlert, 'warn',
          '后端接口未就绪，聊天功能可能不可用。请确认服务已启动后刷新页面。');
      });

    /* tools */
    fetch('/api/tools')
      .then(function (r) {
        if (!r.ok) { throw new Error('HTTP ' + r.status); }
        return r.json();
      })
      .then(renderTools)
      .catch(function () {
        clear(elToolsBody);
        elToolsBody.appendChild($('p', 'muted', '工具清单读取失败。'));
      });

    /* demo orders —— 失败也要渲染欢迎卡片（用占位订单号） */
    fetch('/api/demo/orders')
      .then(function (r) {
        if (!r.ok) { throw new Error('HTTP ' + r.status); }
        return r.json();
      })
      .then(function (data) {
        cachedOrders = Array.isArray(data) ? data : [];
        ordersFailed = false;
        clear(elMessages);
        renderWelcome(cachedOrders, false);
      })
      .catch(function () {
        cachedOrders = [];
        ordersFailed = true;
        clear(elMessages);
        renderWelcome([], true);
      });

    elInput.focus();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
