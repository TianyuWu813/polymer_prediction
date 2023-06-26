class FullAttention(nn.Module):  # Transformer
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module): #ProbSparse Attention
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn

class LogSparse_Attention(nn.Module):
    def __init__(self, n_head, n_embd, win_len, scale, q_len, sub_len, sparse=None, attn_pdrop=0.1, resid_pdrop=0.1):
        super(Attention, self).__init__()

        if(sparse):
            print('Activate log sparse!')
            mask = self.log_mask(win_len, sub_len)
        else:
            mask = torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)

        self.register_buffer('mask_tri', mask)

        self.n_head = n_head
        self.split_size = n_embd * self.n_head
        self.scale = scale
        self.q_len = q_len
        self.query_key = nn.Conv1d(n_embd, n_embd * n_head * 2, self.q_len)
        self.value = Conv1D(n_embd * n_head, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_embd * self.n_head)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def log_mask(self, win_len, sub_len):
        mask = torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i, sub_len, win_len)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, win_len):
        """
        Remark:
        1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
            should deal with CUDA kernel, which we haven't implemented yet.
        2 . Our default setting here use Local attention and Restart attention.
        3 . For index-th row, if its past is smaller than the number of cells the last
            cell can attend, we can allow current cell to attend all past cells to fully
            utilize parallel computing in dense matrices with sparse multiplication."""
        log_l = math.ceil(np.log2(sub_len))
        mask = torch.zeros((win_len), dtype=torch.float)
        if((win_len // sub_len) * 2 * (log_l) > index):
            mask[:(index + 1)] = 1
        else:
            while(index >= 0):
                if((index - log_l + 1) < 0):
                    mask[:index] = 1
                    break
                mask[index - log_l + 1:(index + 1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2**i
                    if((index - new_index) <= sub_len and new_index >= 0):
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def attn(self, query: torch.Tensor, key, value: torch.Tensor, activation="Softmax"):
        activation = activation_dict[activation](dim=-1)
        pre_att = torch.matmul(query, key)
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        mask = self.mask_tri[:, :, :pre_att.size(-2), :pre_att.size(-1)]
        pre_att = pre_att * mask + -1e9 * (1 - mask)
        pre_att = activation(pre_att)
        pre_att = self.attn_dropout(pre_att)
        attn = torch.matmul(pre_att, value)

        return attn

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):

        value = self.value(x)
        qk_x = nn.functional.pad(x.permute(0, 2, 1), pad=(self.q_len - 1, 0))
        query_key = self.query_key(qk_x).permute(0, 2, 1)
        query, key = query_key.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        attn = self.attn(query, key, value)
        attn = self.merge_heads(attn)
        attn = self.c_proj(attn)
        attn = self.resid_dropout(attn)
        return attn

class Conv1D(nn.Module):
    def __init__(self, out_dim, rf, in_dim):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.out_dim = out_dim
        if rf == 1:
            w = torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(out_dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.out_dim,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x

class LSHAttention(nn.Module):
    def __init__( self,
                  dropout = 0.,
                  bucket_size = 64,
                  n_hashes = 8,
                  causal = False,
                  allow_duplicate_attention = True,
                  attend_across_buckets = True,
                  rehash_each_round = True,
                  drop_for_hash_rate = 0.0,
                  random_rotations_per_head = False,
                  return_attn = False):
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.bucket_size = bucket_size

        self.n_hashes = n_hashes

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn

        # cache buckets for reversible network, reported by authors to make Reformer work at depth
        self._cache = {}

    @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)

        random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)

        if self._rehash_each_round:
            # rotated_vectors size [batch,n_hash,seq_len,buckets]
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            buckets = torch.argmax(rotated_vecs, dim=-1)
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = torch.squeeze(rotated_vecs, 1)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            # buckets size [batch size, seq_len, buckets]
            buckets = buckets[... , -self.n_hashes:].transpose(1, 2)

        # buckets is now (self.n_hashes, seq_len). Next we add offsets so that
        # bucket numbers from different hashing rounds don't overlap.
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
        buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
        return buckets

    def forward(self, qk, v, query_len = None, input_mask = None, input_attn_mask = None, pos_emb = None, **kwargs):
        batch_size, seqlen, dim, device = *qk.shape, qk.device

        query_len = default(query_len, seqlen)
        is_reverse = kwargs.pop('_reverse', False)
        depth = kwargs.pop('_depth', None)

        assert seqlen % (self.bucket_size * 2) == 0, f'Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}'

        n_buckets = seqlen // self.bucket_size
        buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training)

        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen

        total_hashes = self.n_hashes

        ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_t = seqlen * buckets + (ticker % seqlen)
        buckets_and_t = buckets_and_t.detach()

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sticker.sort(dim=-1)
        del ticker

        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()

        if exists(pos_emb):
            qk = apply_rotary_pos_emb(qk, pos_emb)

        st = (sticker % seqlen)
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = total_hashes * n_buckets
        bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

        # Dot-product attention.
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)
        masked_value = max_neg_value(dots)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seqlen - input_attn_mask.shape[-1], 0, seqlen - input_attn_mask.shape[-2]), value=True)
            dot_attn_indices = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            input_attn_mask = input_attn_mask.reshape(batch_size, -1)
            dot_attn_indices = dot_attn_indices.reshape(batch_size, -1)
            mask = input_attn_mask.gather(1, dot_attn_indices).reshape_as(dots)
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), value=True)
            mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            if seqlen > query_len:
                mask = mask & (bkv_t[:, :, None, :] < query_len)
            dots.masked_fill_(mask, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t // seqlen, (batch_size, chunk_size, -1))
            bkv_buckets = look_one_back(bkv_buckets)
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots.masked_fill_(bucket_mask, masked_value)
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % chunk_size
            if not self._attend_across_buckets:
                locs1 = buckets * chunk_size + locs1
                locs2 = buckets * chunk_size + locs2
            locs = torch.cat([
                torch.reshape(locs1, (batch_size, total_hashes, seqlen)),
                torch.reshape(locs2, (batch_size, total_hashes, seqlen)),
            ], 1).permute((0, 2, 1))

            slocs = batched_index_select(locs, st)
            b_locs = torch.reshape(slocs, (batch_size, chunk_size, -1, 2 * total_hashes))

            b_locs1 = b_locs[:, :, :, None, :total_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, total_hashes))
            bq_locs = torch.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(total_hashes * batch_size))
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == dots.shape
            dots = dots - torch.log(dup_counts + 1e-9)
            del dup_counts

        # Softmax.
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type_as(dots)
        dropped_dots = self.dropout(dots)

        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,))

        # unsort logits
        o = batched_index_select(so, undo_sort)
        logits = slogits.gather(1, undo_sort)

        o = torch.reshape(o, (batch_size, total_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, total_hashes, seqlen, 1))

        if query_len != seqlen:
            query_slice = (slice(None), slice(None), slice(0, query_len))
            o, logits = o[query_slice], logits[query_slice]

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)

        attn = torch.empty(0, device=device)

        # return unsorted attention weights
        if self._return_attn:
            attn_unsort = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            attn_unsort = attn_unsort.view(batch_size * total_hashes, -1).long()
            unsorted_dots = torch.zeros(batch_size * total_hashes, seqlen * seqlen, device=device)
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, total_hashes, seqlen, seqlen)
            attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)

        # return output, attention matrix, and bucket distribution
        return out, attn, buckets

class Single-Headed Attention(nn.Module):
    def __init__(self, nhid, q=True, k=False, v=False, r=False, heads=1, dropout=None):
        super().__init__()
        self.qs = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.ks = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.vs = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.qkvs = nn.Parameter(torch.zeros(size=(1, 3, nhid), dtype=torch.float))
        self.heads = heads
        self.nhid = nhid
        assert nhid % self.heads == 0, 'Heads must divide vector evenly'
        self.drop = nn.Dropout(dropout) if dropout else None
        self.gelu = GELU()
        self.q = nn.Linear(nhid, nhid) if q else None
        self.qln = LayerNorm(nhid, eps=1e-12)
        self.k = nn.Linear(nhid, nhid) if k else None
        self.v = nn.Linear(nhid, nhid) if v else None
        self.r = nn.Linear(2 * nhid, nhid) if r else None
        self.r_gate = nn.Parameter(torch.ones(size=(1, 1, nhid), dtype=torch.float))
        self.vq = None
        self.vq = Overparam(nhid)
        #from fastai.text.models import QRNNLayer
        #self.vq = QRNNLayer(input_size=nhid, hidden_size=nhid, save_prev_x=False, zoneout=0, window=1, output_gate=False, batch_first=False)
        self.vq_collapsed = False

    def vq_collapse(self):
        vs = torch.sigmoid(self.vs)
        #vs, _ = self.vq(vs)
        vs = self.vq(vs)
        self.vs.data = vs.data
        self.vq = None
        self.vq_collapsed = True

    def forward(self, query, key, value, attn_mask=None, batch_first=False, **kwargs):
        # tanh on the value allows us to flip the polarity of the output, helping use the full range
        # Discovered accidentally when I used QRNN_with_tanh_output(sigmoid(vs))
        #qs, ks, vs = torch.sigmoid(self.qs), torch.sigmoid(self.ks), self.vs
        qs, ks, vs = torch.sigmoid(self.qs), torch.sigmoid(self.ks), torch.sigmoid(self.vs)
        #qs, ks, vs = self.qs, self.ks, self.vs
        #vs = torch.tanh(self.vs)
        if self.vq:
            #vs, _ = self.vq(vs)
            vs = self.vq(vs)
            #qs, ks, vs = [x.reshape((1, 1, -1)) for x in self.vq(torch.sigmoid(self.qkvs))[0, :]]
        elif self.vq_collapsed:
            vs = self.vs
        #qs, ks, vs = self.qs, self.ks, self.vs
        #q = qs * query
        #if self.q: query = self.q(query)
        if self.q:
            query = self.q(query)
            query = self.qln(query.float())
        if self.k: key = self.k(key)
        if self.v: value = self.v(value)
        # This essentially scales everything to zero to begin with and then learns from there
        #q, k, v = self.qs * query, self.ks * key, self.vs * value
        q, k, v = qs * query, ks * key, vs * value
        #q, k, v = query, key, vs * value
        #q, k, v = qs * query, ks * key, value
        #k, v = ks * key, vs * value
        #q, k, v = query, key, value
        if self.drop:
            # We won't apply dropout to v as we can let the caller decide if dropout should be applied to the output
            # Applying dropout to q is equivalent to the same mask on k as they're "zipped"
            #q, k, v = self.drop(q), k, v
            q, k, v = self.drop(q), k, self.drop(v)

        original_q = q

        if not batch_first:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        batch_size, query_len, nhid = q.size()
        assert nhid == self.nhid
        key_len = k.size(1)
        ###
        dim = self.nhid // self.heads
        q = q.view(batch_size, query_len, self.heads, dim).transpose(1, 2)
        k, v = [vec.view(batch_size, key_len, self.heads, dim).transpose(1, 2) for vec in [k, v]]

        mix, focus = attention(q, k, v, dropout=self.drop, attn_mask=attn_mask, **kwargs)
        mix = mix.transpose(1, 2).contiguous().view(batch_size, -1, self.nhid)
        if not batch_first:
            mix = mix.transpose(0, 1)

        if self.r:
            # The result should be transformed according to the query
            r = torch.cat([mix, original_q], dim=-1)
            if self.drop: r = self.drop(r)
            r = self.gelu(self.r(r))
            mix = torch.sigmoid(self.r_gate) * mix + r
            # BUG: This does _nothing_ as mix isn't set to r ...
            # But ... I got good results with this ... so ...
            # Let's leave it as is for right now ...
            # This does imply that I don't necessarily need complex post mixing ops

        return mix, focus

class MultiheadLinearAttention(nn.Module):
    """Based on "Linformer: Self-Attention with Linear Complexity" (https://arxiv.org/abs/2006.04768)"""
    def __init__(self, embed_dim, project_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scale = 1 / math.sqrt(self.head_dim)
        self.query_embed_linear = nn.Linear(embed_dim, embed_dim)
        self.key_embed_linear = nn.Linear(embed_dim, embed_dim)
        self.value_embed_linear = nn.Linear(embed_dim, embed_dim)
        self.key_project_linear = nn.Linear(embed_dim, num_heads * project_dim)
        self.value_project_linear = nn.Linear(embed_dim, num_heads * project_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1:
                nn.init.constant_(p, 0.)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        tgt_len = query.size(0)
        src_len = key.size(0)
        bs = query.size(1)
        q = self.query_embed_linear(query).view(tgt_len, bs * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.key_embed_linear(key).view(src_len, bs * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.value_embed_linear(value).view(src_len, bs * self.num_heads, self.head_dim).transpose(0, 1)
        e = self.key_project_linear(key).view(src_len, bs * self.num_heads, self.project_dim).permute(1, 2, 0)
        f = self.value_project_linear(value).view(src_len, bs * self.num_heads, self.project_dim).permute(1, 2, 0)
        attn = self.scale * q @ (e @ k).transpose(1, 2)
        # masking code from PyTorch MultiheadAttention source code
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float('-inf'))
            else:
                attn += attn_mask
        if key_padding_mask is not None:
            attn = attn.view(bs, self.num_heads, tgt_len, self.project_dim)
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn = attn.view(bs * self.num_heads, tgt_len, self.project_dim)
        attn = F.dropout(F.softmax(attn, dim=-1), p=self.dropout, training=self.training)
        out = attn @ (f @ v)
        out = self.out_linear(out.transpose(0, 1).contiguous().view(tgt_len, bs, self.embed_dim))
        if need_weights:
            attn = attn.view(bs, self.num_heads, tgt_len, self.project_dim).sum(dim=1) / self.num_heads
            return out, attn
        else:
            return out, None

class SparseMultiheadAttention(nn.Module):
    """Simple sparse multihead attention using a limited attention span"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, attn_span=50):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn_span = attn_span
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.query_ff = nn.Linear(embed_dim, embed_dim)
        self.key_ff = nn.Linear(embed_dim, embed_dim)
        self.value_ff = nn.Linear(embed_dim, embed_dim)
        self.out_ff = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, key, value, **kwargs):
        # pytorch sparse tensors still under active development, so expect changes soon
        # for example, sparse batch matrix multiplication is not currently supported
        # TODO add support for masks
        m = query.size(0)
        n = key.size(0)
        if key.size(0) != value.size(0):
            raise RuntimeError("key and value must have same length")
        query = self.query_ff(query).view(m, -1, self.head_dim).transpose(0, 1)
        key = self.key_ff(key).view(n, -1, self.head_dim).transpose(0, 1)
        value = self.value_ff(value).view(n, -1, self.head_dim).transpose(0, 1)
        rows = torch.arange(m, device=query.device).repeat(2 * self.attn_span + 1, 1).transpose(0, 1).flatten()
        cols = torch.cat([torch.arange(i - self.attn_span, i + self.attn_span + 1, device=query.device) for i in range(n)])
        bounds = (cols >= 0) & (cols < n)
        cols[~bounds] = 0
        idxs = torch.stack([rows, cols])
        vals = (query[:, rows, :] * key[:, cols, :] * bounds.view(1, -1, 1)).sum(-1) / math.sqrt(n)
        vals[:, ~bounds] = -float("inf")
        vals = torch.dropout(torch.softmax(vals.view(-1, n, 2 * self.attn_span + 1), dim=-1), self.dropout, self.training).view(-1, idxs.size(1))
        attn_matrix = [torch.sparse.FloatTensor(idxs[:, bounds], val[bounds], (m, n)) for val in vals]
        out = self.out_ff(torch.stack([torch.sparse.mm(attn, val) for attn, val in zip(attn_matrix, value)]).transpose(0, 1).contiguous().view(n, -1, self.embed_dim))
        return out, attn_matrix

# Use this to replace Transformer MultiheadAttention with SparseMultiheadAttention
def replace_modules(model, target, replacement, *args, **kwargs):
    for attr in dir(model):
        module = getattr(model, attr)
        if type(module) is target:
            setattr(model, attr, replacement(*args, **kwargs))
    for child in model.children():
        replace_modules(child, target, replacement, *args, **kwargs)

