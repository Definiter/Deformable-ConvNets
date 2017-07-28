use strict;
use warnings;
use AI::MXNet qw(mx);
use PDL;
use Test::More tests => 37;

sub test_rnn
{
    my $cell = mx->rnn->RNNCell(100, prefix=>'rnn_');
    my ($outputs) = $cell->unroll(3, input_prefix=>'rnn_');
    $outputs = mx->sym->Group($outputs);
    is_deeply([sort keys %{$cell->params->_params}], ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']);
    is_deeply($outputs->list_outputs(), ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']);
    my (undef, $outs, undef) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

sub test_lstm
{
    my $cell = mx->rnn->LSTMCell(100, prefix=>'rnn_', forget_bias => 1);
    my($outputs) = $cell->unroll(3, input_prefix=>'rnn_');
    $outputs = mx->sym->Group($outputs);
    is_deeply([sort keys %{$cell->params->_params}], ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']);
    is_deeply($outputs->list_outputs(), ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']);
    my (undef, $outs, undef) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

sub test_lstm_forget_bias
{
    my $forget_bias = 2;
    my $stack = mx->rnn->SequentialRNNCell();
    $stack->add(mx->rnn->LSTMCell(100, forget_bias=>$forget_bias, prefix=>'l0_'));
    $stack->add(mx->rnn->LSTMCell(100, forget_bias=>$forget_bias, prefix=>'l1_'));

    my $dshape = [32, 1, 200];
    my $data   = mx->sym->Variable('data');

    my ($sym) = $stack->unroll(1, inputs => $data, merge_outputs => 1);
    my $mod = mx->mod->Module($sym, context => mx->cpu(0));
    $mod->bind(data_shapes=>[['data', $dshape]]);

    $mod->init_params();
    my ($bias_argument) = grep { /i2h_bias$/ } @{ $sym->list_arguments };
    my $f = zeros(100);
    my $expected_bias = $f->glue(0, $forget_bias * ones(100), zeros(200));
    ok(
        ((($mod->get_params())[0]->{$bias_argument}->aspdl - $expected_bias)->abs < 1e-07)->all
    );
}

sub test_gru
{
    my $cell = mx->rnn->GRUCell(100, prefix=>'rnn_');
    my($outputs) = $cell->unroll(3, input_prefix=>'rnn_');
    $outputs = mx->sym->Group($outputs);
    is_deeply([sort keys %{$cell->params->_params}], ['rnn_h2h_bias', 'rnn_h2h_weight', 'rnn_i2h_bias', 'rnn_i2h_weight']);
    is_deeply($outputs->list_outputs(), ['rnn_t0_out_output', 'rnn_t1_out_output', 'rnn_t2_out_output']);
    my (undef, $outs, undef) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

sub test_stack
{
    my $cell = mx->rnn->SequentialRNNCell();
    for my $i (0..4)
    {
        $cell->add(mx->rnn->LSTMCell(100, prefix=>"rnn_stack${i}_"));
    }
    my ($outputs) = $cell->unroll(3, input_prefix=>'rnn_');
    $outputs = mx->sym->Group($outputs);
    my %params = %{ $cell->params->_params };
    for my $i (0..4)
    {
        ok(exists $params{"rnn_stack${i}_h2h_weight"});
        ok(exists $params{"rnn_stack${i}_h2h_bias"});
        ok(exists $params{"rnn_stack${i}_i2h_weight"});
        ok(exists $params{"rnn_stack${i}_i2h_bias"});
    }
    is_deeply($outputs->list_outputs(), ['rnn_stack4_t0_out_output', 'rnn_stack4_t1_out_output', 'rnn_stack4_t2_out_output']);
    my (undef, $outs, undef) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

sub test_bidirectional
{
    my $cell = mx->rnn->BidirectionalCell(
        mx->rnn->LSTMCell(100, prefix=>'rnn_l0_'),
        mx->rnn->LSTMCell(100, prefix=>'rnn_r0_'),
        output_prefix=>'rnn_bi_'
    );
    my ($outputs) = $cell->unroll(3, input_prefix=>'rnn_');
    $outputs = mx->sym->Group($outputs);
    is_deeply($outputs->list_outputs(), ['rnn_bi_t0_output', 'rnn_bi_t1_output', 'rnn_bi_t2_output']);
    my (undef, $outs, undef) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 200], [10, 200], [10, 200]]);
}

sub test_unfuse
{
    my $cell = mx->rnn->FusedRNNCell(
        100, num_layers => 1, mode => 'lstm',
        prefix => 'test_', bidirectional => 1
    )->unfuse;
    my ($outputs) = $cell->unroll(3, input_prefix=>'rnn_');
    $outputs = mx->sym->Group($outputs);
    is_deeply($outputs->list_outputs(), ['test_bi_lstm_0t0_output', 'test_bi_lstm_0t1_output', 'test_bi_lstm_0t2_output']);
    my (undef, $outs, undef) = $outputs->infer_shape(rnn_t0_data=>[10,50], rnn_t1_data=>[10,50], rnn_t2_data=>[10,50]);
    is_deeply($outs, [[10, 200], [10, 200], [10, 200]]);
}

sub test_zoneout
{
    my $cell = mx->rnn->ZoneoutCell(
        mx->rnn->RNNCell(100, prefix=>'rnn_'),
        zoneout_outputs => 0.5,
        zoneout_states  => 0.5
    );
    my $inputs = [map { mx->sym->Variable("rnn_t${_}_data") } 0..2];
    my ($outputs) = $cell->unroll(3, inputs => $inputs);
    $outputs = mx->sym->Group($outputs);
    my (undef, $outs) = $outputs->infer_shape(rnn_t0_data=>[10, 50], rnn_t1_data=>[10, 50], rnn_t2_data=>[10, 50]);
    is_deeply($outs, [[10, 100], [10, 100], [10, 100]]);
}

test_rnn();
test_lstm();
test_lstm_forget_bias();
test_gru();
test_stack();
test_bidirectional();
test_unfuse();
test_zoneout();
